# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
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
from open_clip import create_model_from_pretrained, get_tokenizer ,create_model_and_transforms
from PIL import Image
from transformers import AutoFeatureExtractor, SwinForImageClassification, AutoImageProcessor, AutoModelForImageClassification




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
            "height": image_height,
            "name": image_name,
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

"""
指定数据集名称，数据集路径，指定json路径,
这里数据集文件夹都放在../datasets/下，仅修改dataset_name即可
"""
dataset_name = 'dataset1' # FISH | clipart1k | ArTaxOr | dataset1 | dataset2 | dataset3 ,后三个是不开放数据集
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
if dataset_name == 'dataset3':
    damage_types = [
    "dent", 
    "scratch", 
    "crack", 
    "shattered glass", 
    "broken lamp",
    "flat tire", 
    ]
    clip_text = [f"a photo of a {damage}" for damage in damage_types]
    TEXT_PROMPT = catgories # 修改dino字符串, 注意改完后要重新搞个cat2id字典，因为最后得分sclae要用

else:
    clip_text = [f"a photo of a {category}" for category in category_names]



results = []
counter = 0
for image in tqdm(images, desc=f"正在推理数据集-{dataset_name}_{shot}shot"): #rgb
    counter += 1
    img_path = f'../datasets/dataset1/test/{image["name"]}'
    im = cv2.imread(img_path)
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.META_ARCHITECTURE = "OpenSetDetectorWithExamples"  # 新增配置
    cfg.MODEL.OPENSET.ENABLED = True  # 启用开放集检测
    cfg.MODEL.OPENSET.EXAMPLES_PATH = "fewshot_examples.pkl"  # 示例特征路径
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    # import ipdb; ipdb.set_trace()
    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Save the result as an image file
    output_file = "output_image.jpg"  # 指定输出文件名
    cv2.imwrite(output_file, out.get_image()[:, :, ::-1])  # 保存图像
    print(f"Output image saved to {output_file}")
    boxes = outputs['instances'].pred_boxes
    class_ids = outputs["instances"].pred_classes.tolist()  # 输出示例：[0, 23, 5]
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = metadata.thing_classes  # 获取类别名称列表
    pred_class_names = [class_names[i] for i in class_ids]  # 示例：['person', 'bear', 'bus']
    import ipdb;ipdb.set_trace()

    """ dataset2上直接推理"""
    # 添加dataset2处理，直接画图和返回
    for bbox, logit, phrase in zip(xywh, logits,phrases):
        result = {
            "image_id": image["image_id"],
            "category_id": category2id[phrase], # 改为id
            # "bbox": bbox.tolist(),
            "bbox": [int(x) for x in bbox.tolist()],
            "score": float(logit)
        }
        results.append(result)
    annotated_frame = annotate(image_source=image['image'][0], boxes=boxes, logits=logits, phrases=phrases)    
    output_dir = f"../result/{dataset_name}_{shot}/images" # 图像保存路径
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{image['image_id']}.jpg", annotated_frame)

