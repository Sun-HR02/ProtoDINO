from groundingdino.util.inference import load_model, load_image, predict, annotate,GroundingDINOVisualizer, predict_with_features
import cv2
import os
from tqdm import tqdm 
from torchvision.ops import box_convert
import json
from typing import Tuple, List
import torch
import numpy as np
from config import ConfigurationManager, DataConfig, ModelConfig
import matplotlib.pyplot as plt
import groundingdino.datasets.transforms as T
import pickle
from transformers import CLIPProcessor, CLIPModel





# 先从标签获取bbox值和图像
# 从json中读取信息 
# 从prompt中判断phrases的id
def get_category_ids(phrases, text_prompt):
    # 将 TEXT_PROMPT 分词
    prompt_phrases = [p.strip() for p in text_prompt.split('.') if p.strip()]
    # 查找每个 phrase 在 prompt_phrases 中的索引
    category_ids = []
    for phrase in phrases:
        if phrase in prompt_phrases:
            category_ids.append(prompt_phrases.index(phrase)+1)
        else:
            category_ids.append(-1)  # 如果 phrase 不在 TEXT_PROMPT 中，返回 -1
    return category_ids

def load_images_from_json(dataset_path,json_data, dataset_mode) -> List[Tuple[np.array, torch.Tensor]]:
    images = []
    if dataset_mode == 'test':
        datapath = 'test'
    else:
        datapath = 'train'
    # 获取图片元信息，并读取图片做变换
    for image_info in json_data['images']:
        image_name = image_info['file_name']
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        image_path = os.path.join(dataset_path, datapath,image_name)
        # images.append([image_id,load_image(image_path), image_width, image_height])
        images.append({
            "image_id": image_id,
            "image": load_image(image_path),
            "width": image_width,
            "height": image_height
        })
    return images
# 从json中统计category并生成prompt
def load_categories_from_json(json_data) -> List[str]:
# 从 categories 中提取所有类名，并根据 id 排序
    categories = sorted(json_data['categories'], key=lambda x: x['id'])
    category_names = [category['name'] for category in categories]
    category_string = ' . '.join(category_names) + ' .'
    category_id_dict = {category['name']: category['id'] for category in categories}
    id_to_category = {category['id']: category['name'] for category in categories}  # 新增反向映射

    return category_string,category_id_dict,id_to_category

def load_annotations_from_json(json_data) -> List[dict]:
    """从JSON数据中提取完整标注信息
    Returns:
        包含完整标注信息的列表，每个元素包含：
        - image_id: 对应图片ID
        - category_id: 类别ID 
        - bbox: [x, y, width, height]
    """
    return [
        {
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"]
        } 
        for ann in json_data['annotations']
    ]



"""
指定数据集名称，数据集路径，指定json路径,
这里数据集文件夹都放在../datasets/下，仅修改dataset_name即可
"""
dataset_name = 'dataset3' # FISH | clipart1k | ArTaxOr | dataset1 | dataset2 | dataset3 ,后三个是不开放数据集
dataset_mode = '10_shot' # test | ( 1_shot | 5_shot | 10_shot | train ),括号里的不参与最终推理
dataset_path = f'../datasets/{dataset_name}/'
json_path = os.path.join(dataset_path, f'annotations/{dataset_mode}.json') 


# 获取config
config_path="configs/test_config.yaml" # 注意，获取中心用的是test_config
data_config, model_config, training_config = ConfigurationManager.load_config(config_path) # 把配置都读了，但是只用了model_config
print('加载模型.....')
model = load_model(model_config,training_config.use_lora)
clip_model = CLIPModel.from_pretrained("../../data/models/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("../../data/models/clip-vit-large-patch14")
# 打印模型结构
# print("\n" + "="*50 + " 模型结构 " + "="*50)
# print(model)
# print("="*120 + "\n")

# 保存模型结构到文件
def save_model_structure(model, filename="model_structure.txt"):
    with open(filename, "w") as f:
        # 基础信息
        f.write(f"Model Class: {model.__class__.__name__}\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write("\n" + "="*50 + " Detailed Structure " + "="*50 + "\n")
        
        # 递归遍历所有子模块
        def parse_module(module, indent=0, parent_name=""):
            lines = []
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                lines.append(" " * indent + f"({full_name}): {child._get_name()}")
                if list(child.named_children()):  # 如果有子模块
                    lines.extend(parse_module(child, indent + 4, full_name))
            return lines
        
        structure_lines = parse_module(model)
        f.write("\n".join(structure_lines))

# save_model_structure(model, f"{dataset_name}_model_structure.txt")

BOX_TRESHOLD = 0.3 
TEXT_TRESHOLD = 0

# 获取注释中的所有种类并构建文本字符串
with open(json_path, 'r') as f: # 读取json文件
        json_data = json.load(f)
images = load_images_from_json(dataset_path, json_data,dataset_mode) # JSON --> images
catgories,category2id,id2category = load_categories_from_json(json_data) # JSON --> categories
annotations = load_annotations_from_json(json_data)
# 在加载images后立即构建字典
image_dict = {img["image_id"]: img for img in images}
# 直接通过字典访问（替代函数）
TEXT_PROMPT = catgories
category_names = [category.strip() for category in catgories.split('.') if category.strip()]
num_of_categories = len(category_names)
features_of_categories = []
features_of_categories_global = []

counter = 0
# grounding_dino提取bbox对应的特征，可以是文本字符串融合过
for ann in annotations: #predict 函数没有支持并行，所以这里只能逐条推理
    counter += 1
    # 图像根据bbox标签裁剪
    # 根据bbox提取图像中的部分数据
    # 将xywh格式转换为x_min, y_min, x_max, y_max格式
    x_min, y_min, width, height = ann['bbox']
    x_max = x_min + width
    y_max = y_min + height

    # 确保坐标是整数
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    image = image_dict[ann['image_id']]
    cropped_image = image['image'][0][y_min:y_max, x_min:x_max] #RGB
    image_masked = image['image'][0]  # RGB
    category = id2category[ann['category_id']] # 标签类别文本
    

    # if cropped_image.shape[-1] == 3:  # 检查是否是 3 通道图像
    #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

    # if image_masked.shape[-1] == 3:  # 检查是否是 3 通道图像
    #     image_masked = cv2.cvtColor(image_masked, cv2.COLOR_RGB2BGR)

    # 先处理局部图
    inputs = clip_processor(images=cropped_image, return_tensors="pt")
    pixel_values = inputs.pixel_values  # 自动添加批次维度 [1, 3, H, W]
    feature_cropped = clip_model.vision_model(pixel_values = pixel_values)
    features_avg_per_box = feature_cropped.pooler_output #[1,1024]
    features_avg_per_box = features_avg_per_box.squeeze(0) #[1024]

    # 处理全局图
    inputs = clip_processor(images=image_masked, return_tensors="pt")
    pixel_values = inputs.pixel_values  # 自动添加批次维度 [1, 3, H, W]
    feature_masked = clip_model.vision_model(pixel_values = pixel_values)
    features_avg_per_img = feature_masked.pooler_output
    features_avg_per_img = features_avg_per_img.squeeze(0) #[1024]
    # import ipdb;ipdb.set_trace()

    """下面是用grounding_dino提取bbox对应的特征，注释掉了"""
    # # 如果原始图像是 RGB 格式，转换为 BGR
    # if cropped_image.shape[-1] == 3:  # 检查是否是 3 通道图像
    #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    # # if counter==2:
    # #     cv2.imwrite(f'cropped.jpg', cropped_image)
    # # 注意图像应经过变换再传入模型
    # transform = T.Compose(
    #     [
    #         T.RandomResize([800], max_size=1333),
    #         T.ToTensor(),
    #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    # )
    # from PIL import Image
    # pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))  # 如果是BGR格式
    # cropped_image, _ = transform(pil_image, None)
    # # 图像拿去预测，得到features
    # boxes, logits, phrases, features = predict_with_features( # 添加了特征返回
    #     model=model,
    #     image=cropped_image,
    #     caption=TEXT_PROMPT,
    #     box_threshold=BOX_TRESHOLD,
    #     text_threshold=TEXT_TRESHOLD,
    #     remove_combined=True,
    #     counter = counter
    # )
    # features_avg_per_box = features.mean(dim=0) # 裁剪后的图如果识别出了多个box，求特征均值



    category_id = ann['category_id']
    features_of_categories.append({
        'category_id': category_id,
        'feature': features_avg_per_box
    })
    category_id = ann['category_id']
    features_of_categories_global.append({
        'category_id': category_id,
        'feature': features_avg_per_img
    })
# 每个类别计算特征平均值，得到一个向量
# 创建按类别分组的字典
category_features = {}
category_features_gloal = {}
for item in features_of_categories:
    category_id = item['category_id']
    if category_id not in category_features:
        category_features[category_id] = []
    category_features[category_id].append(item['feature'])
for item in features_of_categories_global:
    category_id = item['category_id']
    if category_id not in category_features_gloal:
        category_features_gloal[category_id] = []
    category_features_gloal[category_id].append(item['feature'])
# 计算每个类别的平均特征
category_centroids = {}
category_centroids_global = {}
for cat_id, features in category_features.items():
    # 将特征列表转为张量 [N, D]
    features_tensor = torch.stack(features, dim=0)
    # 沿第一个维度取平均 [D]
    mean_feature = torch.mean(features_tensor, dim=0)
    category_centroids[cat_id] = mean_feature
for cat_id, features in category_features_gloal.items():
    # 将特征列表转为张量 [N, D]
    features_tensor = torch.stack(features, dim=0)
    # 沿第一个维度取平均 [D]
    mean_feature = torch.mean(features_tensor, dim=0)
    category_centroids_global[cat_id] = mean_feature


# centroid_tensors = torch.stack(list(category_centroids.values()), dim=0)
print(f"生成 {len(category_centroids)} 个类别的中心向量")
for cat_id, vec in category_centroids.items():
    print(f"类别 {cat_id}: 向量维度 {vec.shape}, NaN 检查: {torch.isnan(vec).any()}")


# 将centroids保存到pkl:
centroids_path = f'{dataset_name}_{dataset_mode}_centroids.pkl'
with open(centroids_path, 'wb') as f:
    pickle.dump(category_centroids, f)
print(f'保存到{dataset_name}_{dataset_mode}_centroids.pkl')

# 将centroids保存到pkl:
centroids_path_global = f'{dataset_name}_{dataset_mode}_centroids_global.pkl'
with open(centroids_path_global, 'wb') as f:
    pickle.dump(category_centroids_global, f)
print(f'保存到{dataset_name}_{dataset_mode}_centroids_global.pkl')