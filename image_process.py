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
def load_images_from_json(dataset_path,json_data) -> List[Tuple[np.array, torch.Tensor]]:
    images = []

    # 通过datapath区分两个图片文件夹，仅train mode下会用train下图片
    # 获取图片元信息，并读取图片做变换
    for image_info in json_data['images']:
        image_name = image_info['file_name']
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        image_path = os.path.join(dataset_path,image_name)
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
    return category_string,category_id_dict

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

def is_bbox_overlap(bbox1, bbox2):
    """
    判断两个左下角坐标系xywh格式的边界框是否完全不相交
    Args:
        bbox1 (list/np.ndarray): [x, y_bottom, width, height]
        bbox2 (list/np.ndarray): [x, y_bottom, width, height]
    Returns:
        bool: 完全不相交时返回False，否则返回True
    """
    # 解包坐标参数
    x1, y_bottom1, w1, h1 = bbox1
    x2, y_bottom2, w2, h2 = bbox2
    # 转换为标准坐标系
    box1_top = y_bottom1 - h1
    box2_top = y_bottom2 - h2
    
    # 计算水平投影和垂直投影是否有交集
    horizontal_overlap = not (x1 + w1 <= x2 or x2 + w2 <= x1)
    vertical_overlap = not (y_bottom1 <= box2_top or y_bottom2 <= box1_top)
    
    return horizontal_overlap and vertical_overlap

def mask_non_overlap_areas(image: np.ndarray, 
                          detect_boxes: np.ndarray, 
                          label_box: list) -> np.ndarray:
    """
    将不与标签框重叠的检测框区域涂黑
    Args:
        image (np.ndarray): 原始BGR格式图像数组，形状[H,W,3]
        detect_boxes (np.ndarray): 检测到的边界框数组，形状[N,4]，格式为xywh(左下角坐标系)
        label_box (list): 标签边界框，格式为xywh(左下角坐标系)
    Returns:
        np.ndarray: 处理后的图像
    """
    # 创建图像副本避免修改原图
    masked_image = image.copy()
    height, width = image.shape[:2]
    print('处理图像..')
    for box in detect_boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height        
        # 确保坐标是整数
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        
        # 跳过无效区域
        if x_max <= x_min or y_max <= y_min:
            continue
            
        # 判断是否与标签框重叠
        if not is_bbox_overlap(box, label_box):
            # 将区域涂黑（OpenCV使用BGR格式）
            masked_image[y_min:y_max, x_min:x_max] = 0
            
    return masked_image

# 读取图像
dataset_path = '../datasets/dataset3/train/'
json_file_path = os.path.join('../datasets/dataset3/annotations', '1_shot.json')
output_path = '../datasets/masked/dataset3/'

with open(json_file_path, 'r') as f: # 读取json文件
        json_data = json.load(f)
images = load_images_from_json(dataset_path, json_data) # JSON --> images
anns = load_annotations_from_json(json_data)
catgories,category2id = load_categories_from_json(json_data) # JSON --> categories
TEXT_PROMPT = catgories
# 获取config
config_path="configs/test_config.yaml" # 注意，获取中心用的是test_config
data_config, model_config, training_config = ConfigurationManager.load_config(config_path) # 把配置都读了，但是只用了model_config
print('加载模型.....')
model = load_model(model_config,training_config.use_lora)


for image in images:
    id = image['image_id']
    width = image['width']
    height = image['height']   
    i = image['image']
    # 查找对应注释
    for ann in anns:
        if ann['image_id'] == id:
            bbox_label = ann['bbox'] # 提出标签的bbox
            break
    counter = 0
    # 用grounding_dino推理图像，拿到推理bbox        
    boxes, logits, phrases, features = predict_with_features( # 添加了特征返回
        model=model,
        image=i[1],
        caption=TEXT_PROMPT,
        box_threshold=0.35,
        text_threshold=0,
        remove_combined=True,
        counter = counter
    ) 
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy() * [image['width'], image['height'], image["width"], image['height']]
    i_original = i[0]
    # import ipdb;ipdb.set_trace()
    masked_img = mask_non_overlap_areas(i_original,xywh,bbox_label)
    # 如果原始图像是 RGB 格式，转换为 BGR
    if masked_img.shape[-1] == 3:  # 检查是否是 3 通道图像
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_path}/{id}_masked.jpg", masked_img)
    annotated_frame = annotate(image_source=i_original, boxes=boxes, logits=logits, phrases=phrases)    
    cv2.imwrite(f"{output_path}/{id}_inferenced.jpg", annotated_frame)
    annotated_frame = annotate(image_source=i_original, boxes=bbox_label, logits=logits, phrases=phrases)    
    cv2.imwrite(f"{output_path}/{id}_truth.jpg", annotated_frame)

print('done')

# 提取相应bbox



