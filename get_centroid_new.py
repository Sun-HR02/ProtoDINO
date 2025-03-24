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
from open_clip import create_model_from_pretrained, get_tokenizer ,create_model_and_transforms
from transformers import AutoFeatureExtractor, SwinForImageClassification, AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F  # 添加在文件开头的import部分
from PIL import Image
from tqdm import tqdm




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
op_clip_model, _, op_preprocess = create_model_and_transforms('ViT-H-14', pretrained='../../data/models/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin')
op_clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
op_clip_tokenizer = get_tokenizer('ViT-H-14','../../data/models/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin')

if dataset_name == 'dataset3':
    car_img_processor = AutoImageProcessor.from_pretrained("../../data/models/car_damage_detection")
    car_img_model = AutoModelForImageClassification.from_pretrained("../../data/models/car_damage_detection")


# 模型数量超了，把swin换回去
# swin_feature_extractor = AutoFeatureExtractor.from_pretrained("../../data/models/swin-large-patch4-window12-384-in22k")
# swin_model = SwinForImageClassification.from_pretrained("../../data/models/swin-large-patch4-window12-384-in22k")


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
for ann in tqdm(annotations, desc=f"正在构建质心-{dataset_name}_{dataset_mode}"): #predict 函数没有支持并行，所以这里只能逐条推理
    counter += 1

    if dataset_name=='dataset3':
        x_min, y_min, width, height = ann['bbox']
        x_max = x_min + width
        y_max = y_min + height
        # 确保坐标是整数
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        image = image_dict[ann['image_id']]
        cropped_image = image['image'][0][y_min:y_max, x_min:x_max] #RGB
        image_masked = image['image'][0]  # RGB
        category = id2category[ann['category_id']] # 标签类别文本
        pil_image = Image.fromarray(cropped_image) # 局部
        img_processed = car_img_processor(images=pil_image, return_tensors="pt")
        # Make predictions
        car_model_prediction = car_img_model(**img_processed,output_hidden_states = True)
        car_model_hidden = car_model_prediction.hidden_states[-1] # [1,197,768]
        last_hidden_mean = car_model_hidden.mean(dim=1)  # 从 [1, 197,768] -> [1, 768]
        features_avg_per_box = F.normalize(last_hidden_mean, p=2, dim=-1)  # L2归一化
        features_avg_per_box = features_avg_per_box.squeeze(0) # [768]

        pil_image = Image.fromarray(image_masked) # 局部
        img_processed = car_img_processor(images=pil_image, return_tensors="pt")
        # Make predictions
        car_model_prediction = car_img_model(**img_processed,output_hidden_states = True)
        car_model_hidden = car_model_prediction.hidden_states[-1] # [1,197,768]
        last_hidden_mean = car_model_hidden.mean(dim=1)  # 从 [1, 197,768] -> [1, 768]
        features_avg_per_img = F.normalize(last_hidden_mean, p=2, dim=-1)  # L2归一化
        features_avg_per_img = features_avg_per_img.squeeze(0) # [768]

    else:
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
        
        # 先处理局部图
        pil_crop = Image.fromarray(cropped_image)
        pil_crop = op_preprocess(pil_crop).unsqueeze(0)  # 添加预处理并增加batch维度
        image_features = op_clip_model.encode_image(pil_crop)
        features_avg_per_box = F.normalize(image_features, dim=-1).squeeze(0) #[ 1024]

        # pil_crop_proceessed = swin_feature_extractor(images=pil_crop, return_tensors="pt")
        # swin_output = swin_model(**pil_crop_proceessed, output_hidden_states = True)
        # # 获取最后一层隐藏状态（形状为 [batch_size, sequence_length, hidden_size]）
        # last_hidden = swin_output.hidden_states[-1]
        # # 添加空间维度池化（提升特征质量）
        # last_hidden_mean = last_hidden.mean(dim=1)  # 从 [1, 144, 1536] -> [1, 1536]
        # features_avg_per_box = F.normalize(last_hidden_mean, p=2, dim=-1)  # L2归一化
        # features_avg_per_box = features_avg_per_box.squeeze(0) # [1536]

        # 处理全局图
        pil_img = Image.fromarray(image_masked)
        pil_img = op_preprocess(pil_img).unsqueeze(0)  # 添加预处理并增加batch维度
        image_features = op_clip_model.encode_image(pil_img)
        features_avg_per_img = F.normalize(image_features, dim=-1).squeeze(0)

        # pil_img_proceessed = swin_feature_extractor(images=pil_img, return_tensors="pt")
        # swin_output = swin_model(**pil_img_proceessed, output_hidden_states = True)
        # # 获取最后一层隐藏状态（形状为 [batch_size, sequence_length, hidden_size]）
        # last_hidden = swin_output.hidden_states[-1]
        # # 添加空间维度池化（提升特征质量）
        # last_hidden_mean = last_hidden.mean(dim=1)  # 从 [1, 144, 1536] -> [1, 1536]
        # features_avg_per_img = F.normalize(last_hidden_mean, p=2, dim=-1)  # L2归一化
        # features_avg_per_img = features_avg_per_img.squeeze(0) # [1536]


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