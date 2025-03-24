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
from transformers import CLIPProcessor, CLIPModel
import pickle
import torch.nn.functional as F  # 添加在文件开头的import部分

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
# 从json中读取信息 
def load_images_from_json(dataset_path,json_data, dataset_mode) -> List[Tuple[np.array, torch.Tensor]]:
    images = []
    if dataset_mode == 'train':
        datapath = 'train'
    else:
        datapath = 'test'
    # 通过datapath区分两个图片文件夹，仅train mode下会用train下图片
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


# 获取config
config_path="configs/test_config.yaml"
data_config, model_config, training_config = ConfigurationManager.load_config(config_path) # 把配置都读了，但是只用了model_config
print('加载模型.....')
model = load_model(model_config,training_config.use_lora)
clip_model = CLIPModel.from_pretrained("../../data/models/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("../../data/models/clip-vit-large-patch14")

# 原groundingdino用的参数
BOX_TRESHOLD = 0.3 # DINO筛选阈值
TEXT_TRESHOLD = 0

"""原来使用两个阈值，但效果不好，修改为单阈值"""
RES_THRESHOLD = 0.7
# # CLIP特征阈值
# CLIP_IMAGE_TRESHOLD = 0.8 # 最低局部图像特征阈值
# CLIP_TEXT_TRESHOLD = 0.8 # 最低文本特征阈值

# 最终得分权重 
w_global = 0.25# 全局图像特征
w_partial = 0.35 # 局部图像特征
w_text = 0.4 # clip文本特征

"""
指定数据集名称，数据集路径，指定json路径,
这里数据集文件夹都放在../datasets/下，仅修改dataset_name即可
"""
dataset_name = 'dataset3' # FISH | clipart1k | ArTaxOr | dataset1 | dataset2 | dataset3 ,后三个是不开放数据集
dataset_mode = 'test' # test | ( 1_shot | 5_shot | 10_shot | train ),括号里的不参与最终推理
shot = 1 # 1 | 5 | 10
dataset_path = f'../datasets/{dataset_name}/'
centorid_path = f'{dataset_name}_{shot}_shot_centroids.pkl'
centorid_path_global = f'{dataset_name}_{shot}_shot_centroids_global.pkl'
json_path = os.path.join(dataset_path, f'annotations/{dataset_mode}.json') 


with open(json_path, 'r') as f: # 读取json文件
        json_data = json.load(f)
images = load_images_from_json(dataset_path, json_data,dataset_mode) # JSON --> images
catgories,category2id,id2category = load_categories_from_json(json_data) # JSON --> categories
TEXT_PROMPT = catgories
# 处理 categories 字符串，提取类别名称
category_names = [category.strip() for category in catgories.split('.') if category.strip()]

# 构造 clip_text 列表
clip_text = [f"a photo of a {category}" for category in category_names]


# 读局部质心
with open(centorid_path, 'rb') as f: 
    centroid_features = pickle.load(f)
print("加载后检查：")
for cat_id, vec in centroid_features.items():
    print(f"类别 {cat_id}: 向量维度 {vec.shape}, NaN 检查: {torch.isnan(vec).any()}")

# 读全局质心
with open(centorid_path_global, 'rb') as f: 
    centroid_features_global = pickle.load(f)

results = []
counter = 0
for image in tqdm(images, desc=f"正在推理数据集-{dataset_name}"): #predict 函数没有支持并行，所以这里只能逐条推理
    counter += 1
    # boxes, logits, phrases = predict( # 原来的推理
    #     model=model,
    #     image=image['image'][1], # 用变换后的图片推理
    #     caption=TEXT_PROMPT,
    #     box_threshold=BOX_TRESHOLD,
    #     text_threshold=TEXT_TRESHOLD,
    #     remove_combined=True,
    #     counter = counter
    # )    
    boxes, logits, phrases, features = predict_with_features( # 添加了特征返回,返回了grounding_dino的特征
        model=model,
        image=image['image'][1], # 用变换后的图片推理
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        remove_combined=True,
        counter = counter
    )
    # 获取xywh格式的边界框坐标相对值， 在结合图片大小转为绝对值
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy() * [image['width'], image['height'], image["width"], image['height']]
    # 根据phrase文本的内容，获取类别id，下标从1
    # category_ids = get_category_ids(phrases, TEXT_PROMPT)
    # category_ids = [category2id[phrase] for phrase in phrases]
    # category_ids = []
    # for phrase in phrases:
    #     if phrase not in category2id:
    #         category_ids.append(1)
    #     else: 
    #         category_ids.append(category2id[phrase])
    # # 分类得分
    # scores = logits.numpy()
    # # 将相关信息保存,包括元数据、类别、边界框、得分


    # 读出每一类的质心
    # 验证质心特征格式
    assert isinstance(centroid_features, dict), "质心特征格式应为字典"
    # print(f"加载 {len(centroid_features)} 个类别的质心特征")
    centroid_tensors = torch.stack(list(centroid_features.values()), dim=0)
    centroid_tensors_global = torch.stack(list(centroid_features_global.values()), dim=0)


    """全局特征不需要获取，注释了"""
    # # 获取图片全局特征
    # # 处理全局图
    # image_masked = image['image'][0]
    # if image_masked.shape[-1] == 3:  # 检查是否是 3 通道图像
    #     image_masked = cv2.cvtColor(image_masked, cv2.COLOR_RGB2BGR)
    # inputs = clip_processor(images=image_masked, return_tensors="pt")
    # pixel_values = inputs.pixel_values  # 自动添加批次维度 [1, 3, H, W]
    # feature_image_masked = clip_model.vision_model(pixel_values = pixel_values).pooler_output.squeeze(0) #[1024]
    # # 计算全局质心得分
    # similarities_global = F.cosine_similarity(feature_image_masked.unsqueeze(0), centroid_tensors_global, dim=1)
    # global_similarity = {
    #     cat_id: similarities_global[i].item()  # 转换为Python float类型
    #     for i, cat_id in enumerate(centroid_features_global.keys())
    # }



    # 特征匹配逻辑
    boxes_ret = []
    box_to_paint = []
    category_ids = []
    scores = []
    for box,box_original in zip(xywh,boxes):
        # 将xywh格式转换为x_min, y_min, x_max, y_max格式, 和标签对上
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        # 确保坐标是整数
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        # 添加边界检查
        # x_min = max(0, x_min)
        # y_min = max(0, y_min)
        # x_max = min(image['width'], x_max)
        # y_max = min(image['height'], y_max)
        # 跳过无效区域
        if x_max <= x_min or y_max <= y_min:
            continue
        cropped_image = image['image'][0][y_min:y_max, x_min:x_max]
        # 添加空图像检查
        if cropped_image.size == 0:
            continue
        # if cropped_image.shape[-1] == 3:  # 检查是否是 3 通道图像
        #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)  
        # cv2.imwrite(f'{image["image_id"]}.png', cropped_image)

        # 先计算文本得分
        inputs = clip_processor(text=clip_text, images=cropped_image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        text_similarity = {
            category2id[category_names[i]]: probs[0, i].item()  # 转换为Python float类型
            for i in range(len(category_names))
        }


        # 使用CLIP的处理器进行预处理
        inputs = clip_processor(images=cropped_image, return_tensors="pt")
        pixel_values = inputs.pixel_values  # 自动添加批次维度 [1, 3, H, W]
        feature_cropped = clip_model.vision_model(pixel_values = pixel_values)
        features_avg_per_box = feature_cropped.pooler_output.squeeze(0) 
        # 计算特征向量与局部质心向量之间的余弦相似度 
        img_similarities = F.cosine_similarity(features_avg_per_box.unsqueeze(0), centroid_tensors, dim=1)
        partial_similarity = {
            cat_id: img_similarities[i].item()  # 转换为Python float类型
            for i, cat_id in enumerate(centroid_features.keys())
        }
        # 计算特征向量与全局质心向量之间的余弦相似度
        similarities_global = F.cosine_similarity(features_avg_per_box.unsqueeze(0), centroid_tensors_global, dim=1)
        global_similarity = {
            cat_id: similarities_global[i].item()  # 转换为Python float类型
            for i, cat_id in enumerate(centroid_features_global.keys())
        }
        # 计算总得分
        total_scores = {}
        for cat_id in category2id.values():
            # 获取各类别得分（注意处理可能缺失的键）
            text_score = text_similarity.get(cat_id, 0.0)
            global_score = global_similarity.get(cat_id, 0.0)
            partial_score = partial_similarity.get(cat_id, 0.0)
            
            # 加权求和
            total_scores[cat_id] = (w_text * text_score) + (w_global * global_score) + (w_partial * partial_score)

        # 找到最高得分类别
        best_cat_id = max(total_scores, key=total_scores.get)
        best_score = total_scores[best_cat_id]
        text_score = text_similarity.get(best_cat_id, 0.0)
        global_score = global_similarity.get(best_cat_id, 0.0)
        partial_score = partial_similarity.get(best_cat_id, 0.0)

        # if partial_score > CLIP_IMAGE_TRESHOLD and text_score > CLIP_TEXT_TRESHOLD:
        if best_score >= RES_THRESHOLD:
            box_to_paint.append(box_original)
            boxes_ret.append(box) 
            category_ids.append(best_cat_id)
            scores.append(best_score)


    for bbox, category_id, score in zip(boxes_ret, category_ids, scores):
        result = {
            "image_id": image["image_id"],
            "category_id": category_id,
            "bbox": bbox.tolist(),  # 将 numpy 数组转换为列表
            "score": float(score)
        }
        
        results.append(result)
    # 渲染图片边框
    # 修改后的代码段（行号204-207）
    if len(box_to_paint) > 0:
        # 确保每个元素是numpy数组后再进行堆叠
        box_to_paint = [np.array(b) for b in box_to_paint]
        box_to_paint = torch.from_numpy(np.stack(box_to_paint)).float()
    else:
        box_to_paint = torch.zeros((0, 4), dtype=torch.float32)  # 明确二维形状
    # import ipdb;ipdb.set_trace()

    category_names_paint = [id2category[i] for i in category_ids]
    annotated_frame = annotate(image_source=image['image'][0], boxes=box_to_paint, logits=scores, phrases=category_names_paint)    
    # 保存图像
    output_dir = f"../result/{dataset_name}_{shot}/images" # 图像保存路径
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{image['image_id']}.jpg", annotated_frame)

# 保存对象相关信息
with open(f'../result/{dataset_name}_{shot}/{dataset_name}.json', 'w') as f:
    json.dump(results, f, indent=4)


