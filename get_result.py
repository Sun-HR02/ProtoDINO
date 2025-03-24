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
    return category_string,category_id_dict


# 获取config
config_path="configs/test_config.yaml"
data_config, model_config, training_config = ConfigurationManager.load_config(config_path) # 把配置都读了，但是只用了model_config
print('加载模型.....')
model = load_model(model_config,training_config.use_lora)


clip_model = CLIPModel.from_pretrained("../../data/models/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("../../data/models/clip-vit-large-patch14")

BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0

"""
指定数据集名称，数据集路径，指定json路径,
这里数据集文件夹都放在../datasets/下，仅修改dataset_name即可
"""
dataset_name = 'dataset3' # FISH | clipart1k | ArTaxOr | dataset1 | dataset2 | dataset3 ,后三个是不开放数据集
dataset_mode = 'test' # test | ( 1_shot | 5_shot | 10_shot | train ),括号里的不参与最终推理
dataset_path = f'../datasets/{dataset_name}/'
json_path = os.path.join(dataset_path, f'annotations/{dataset_mode}.json') 


with open(json_path, 'r') as f: # 读取json文件
        json_data = json.load(f)
images = load_images_from_json(dataset_path, json_data,dataset_mode) # JSON --> images
catgories,category2id = load_categories_from_json(json_data) # JSON --> categories
TEXT_PROMPT = catgories
# 处理 categories 字符串，提取类别名称
category_names = [category.strip() for category in catgories.split('.') if category.strip()]

# 构造 clip_text 列表
# clip_text = [f"a photo of a {category}" for category in category_names]
clip_text = [f"{category}" for category in category_names]


results = []
counter = 0
for image in tqdm(images, desc=f"正在推理数据集-{dataset_name}"): #predict 函数没有支持并行，所以这里只能逐条推理
    counter += 1
    boxes, logits, phrases = predict( # 原来的推理
        model=model,
        image=image['image'][1], # 用变换后的图片推理
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        remove_combined=True,
        counter = counter
    )    
    # boxes, logits, phrases, features = predict_with_features( # 添加了特征返回
    #     model=model,
    #     image=image['image'][1], # 用变换后的图片推理
    #     caption=TEXT_PROMPT,
    #     box_threshold=BOX_TRESHOLD,
    #     text_threshold=TEXT_TRESHOLD,
    #     remove_combined=True,
    #     counter = counter
    # )
    
    # 获取xywh格式的边界框坐标相对值， 在结合图片大小转为绝对值
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy() * [image['width'], image['height'], image["width"], image['height']]
    # 根据phrase文本的内容，获取类别id，下标从1
    category_ids = get_category_ids(phrases, TEXT_PROMPT)
    category_ids = [category2id[phrase] for phrase in phrases]
    category_ids = []
    for phrase in phrases:
        if phrase not in category2id:
            category_ids.append(1)
        else: 
            category_ids.append(category2id[phrase])
    # 分类得分
    scores = logits.numpy()
    # 将相关信息保存,包括元数据、类别、边界框、得分


    print_categories = [] # clip分类后，保存类别名
    print_scores = []
    print_boxes = []
    for bbox, category_id, score, box_original in zip(xywh, category_ids, scores, boxes):

        """为了clip 模型进行推理，需要将图片裁剪出部分区域，并使用clip模型进行推理，这里使用bbox进行裁剪"""
        # 根据bbox提取图像中的部分数据
        # 将xywh格式转换为x_min, y_min, x_max, y_max格式
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        # 确保坐标是整数
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        cropped_image = image['image'][0][y_min:y_max, x_min:x_max]
        # 如果原始图像是 RGB 格式，转换为 BGR(发现不转换效果更好，所以就注释掉了)
        if cropped_image.shape[-1] == 3:  # 检查是否是 3 通道图像
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        inputs = clip_processor(text=clip_text, images=cropped_image, return_tensors="pt", padding=True)

        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        # 根据概率输出最可能类别
        predicted_class_idx = clip_text[probs.argmax()]
        clip_score = probs.max()
        # 最后一个单词，如果改提示词可能用得到
        # predicted_class_idx = predicted_class_idx.split()[-1]
        if clip_score > 0.9:
            print_categories.append(predicted_class_idx)
            print_boxes.append(box_original)
            print_scores.append(clip_score)
            result = {
                "image_id": image["image_id"],
                "category_id": category2id[predicted_class_idx],
                "bbox": bbox.tolist(),  # 将 numpy 数组转换为列表
                "score": float(clip_score)
            } 

            # result = {
            #     "image_id": image["image_id"],
            #     "category_id": category_id,
            #     "bbox": bbox.tolist(),  # 将 numpy 数组转换为列表
            #     "score": float(score)
            # }
        
            results.append(result)
    # 渲染图片边框
    if len(print_boxes) > 0:
        # 确保每个元素是numpy数组后再进行堆叠
        print_boxes = [np.array(b) for b in print_boxes]
        print_boxes = torch.from_numpy(np.stack(print_boxes)).float()
    else:
        print_boxes = torch.zeros((0, 4), dtype=torch.float32)  # 明确二维形状
    annotated_frame = annotate(image_source=image['image'][0], boxes=print_boxes, logits=print_scores, phrases=print_categories)     # 用clip后
    # annotated_frame = annotate(image_source=image['image'][0], boxes=boxes, logits=logits, phrases=phrases)    
    # 保存图像
    output_dir = f"../result/{dataset_name}/images" # 图像保存路径
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{image['image_id']}.jpg", annotated_frame)

# 保存对象相关信息
with open(f'../result/{dataset_name}/{dataset_name}.json', 'w') as f:
    json.dump(results, f, indent=4)

