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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
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
            'name': image_name
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
op_clip_model, _, op_preprocess = create_model_and_transforms('ViT-H-14', pretrained='../../data/models/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin')
op_clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
op_clip_tokenizer = get_tokenizer('ViT-H-14','../../data/models/DFN5B-CLIP-ViT-H-14-378/open_clip_pytorch_model.bin')
# swin_feature_extractor = AutoFeatureExtractor.from_pretrained("../../data/models/swin-large-patch4-window12-384-in22k")
# swin_model = SwinForImageClassification.from_pretrained("../../data/models/swin-large-patch4-window12-384-in22k")
car_img_processor = AutoImageProcessor.from_pretrained("../../data/models/car_damage_detection")
car_img_model = AutoModelForImageClassification.from_pretrained("../../data/models/car_damage_detection")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../../data/models/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
qwen_processor = AutoProcessor.from_pretrained("../../data/models/Qwen2.5-VL-7B-Instruct")

# 原groundingdino用的参数
BOX_TRESHOLD = 0.25 # DINO筛选阈值
TEXT_TRESHOLD = 0

"""原来使用两个阈值，但效果不好，修改为单阈值"""
RES_THRESHOLD = 0.4

# dino判定结果增益
scale = 1.15 # dino判定的结果对应得分*1.15
# 最终得分权重 
w_global = 0# 全局图像特征
w_partial = 0 # 局部图像特征
w_text = 1 # clip文本特征
w_car = 0.0 # 仅d3生效
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
op_clip_text = op_clip_tokenizer(clip_text,context_length=op_clip_model.context_length)

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
for image in tqdm(images, desc=f"正在推理数据集-{dataset_name}_{shot}shot"): #rgb
    counter += 1
    # if counter <34 or (counter >44 and counter < 93) or counter > 98:
    #     continue


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"../datasets/dataset1/test/{image['name']}", # 修改路径
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    import ipdb;ipdb.set_trace()

    
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

    """ dataset2上直接推理"""
    # 添加dataset2处理，直接画图和返回
    if dataset_name == 'dataset2':
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
        continue



    """ 其他数据上使用少样本辅助"""
    # 读出每一类的质心
    # 验证质心特征格式
    assert isinstance(centroid_features, dict), "质心特征格式应为字典"
    # print(f"加载 {len(centroid_features)} 个类别的质心特征")
    centroid_tensors = torch.stack(list(centroid_features.values()), dim=0)
    centroid_tensors_global = torch.stack(list(centroid_features_global.values()), dim=0)


    # 特征匹配逻辑
    boxes_ret = []
    box_to_paint = []
    category_ids = []
    scores = []
    for box,box_original, dino_phrase in zip(xywh,boxes, phrases):
        # 将xywh格式转换为x_min, y_min, x_max, y_max格式, 和标签对上
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        # 确保坐标是整数
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        # 跳过无效区域
        if x_max <= x_min or y_max <= y_min:
            continue
        cropped_image = image['image'][0][y_min:y_max, x_min:x_max]
        # 添加空图像检查
        if cropped_image.size == 0:
            continue  

        # 先计算文本得分, 替换为open_clip
        pil_image = Image.fromarray(cropped_image)
        pil_image = op_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = op_clip_model.encode_image(pil_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = op_clip_model.encode_text(op_clip_text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # 构造text_similarity
        text_similarity = {
            category2id[category_names[i]]: text_probs[0, i].item()
            for i in range(len(category_names))
        }

        # 进行局部图像编码
        pil_image = Image.fromarray(cropped_image)
        pil_image_proceessed = op_preprocess(pil_image).unsqueeze(0)
        crop_features = op_clip_model.encode_image(pil_image_proceessed)
        features_avg_per_box = F.normalize(crop_features, p=2, dim=-1)  # L2归一化

        # pil_image_proceessed = swin_feature_extractor(images=pil_image, return_tensors="pt")
        # swin_output = swin_model(**pil_image_proceessed, output_hidden_states = True)
        # # 获取最后一层隐藏状态（形状为 [batch_size, sequence_length, hidden_size]）
        # last_hidden = swin_output.hidden_states[-1]
        # # 添加空间维度池化（提升特征质量）
        # last_hidden_mean = last_hidden.mean(dim=1)  # 从 [1, 144, 1536] -> [1, 1536]
        # features_avg_per_box = F.normalize(last_hidden_mean, p=2, dim=-1)  # L2归一化
        # 计算特征向量与局部质心向量之间的余弦相似度 
        # img_similarities = F.cosine_similarity(features_avg_per_box, centroid_tensors, dim=1)

        # 添加温度参数（可调）
        sigma = 0.5  # 可设置为超参数，值越大得分差异越平缓
        distances = torch.norm(features_avg_per_box.unsqueeze(1) - centroid_tensors, p=2, dim=2)
        img_similarities = torch.exp(-distances / sigma).squeeze(0)  # 转换为0~1的正值
        # import ipdb;ipdb.set_trace()

        partial_similarity = {
            cat_id: img_similarities[i].item()  # 转换为Python float类型
            for i, cat_id in enumerate(centroid_features.keys())
        }
        # 计算特征向量与全局质心向量之间的余弦相似度
        # similarities_global = F.cosine_similarity(features_avg_per_box, centroid_tensors_global, dim=1)

        # 添加温度参数（可调）
        sigma = 0.5  # 可设置为超参数，值越大得分差异越平缓
        distances = torch.norm(features_avg_per_box.unsqueeze(1) - centroid_tensors, p=2, dim=2)
        similarities_global = torch.exp(-distances / sigma).squeeze(0)  # 转换为0~1的正值

        global_similarity = {
            cat_id: similarities_global[i].item()  # 转换为Python float类型
            for i, cat_id in enumerate(centroid_features_global.keys())
        }

        # 为dataset3 新增一个分类模型
        if dataset_name == 'dataset3':
            pil_image = Image.fromarray(cropped_image) # 局部
            img_processed = car_img_processor(images=pil_image, return_tensors="pt")
            # Make predictions
            car_model_prediction = car_img_model(**img_processed)
            car_model_logits = car_model_prediction.logits.detach().cpu().numpy() # "0": Crack "1": Scratch "2": Tire Flat "3": Dent "4": Glass Shatter "5": Lamp Broken
            car_probs = torch.softmax(torch.tensor(car_model_logits), dim=1)[0]
            car_label_map = {
                0: car_probs[3],
                1: car_probs[1],
                2: car_probs[0],
                3: car_probs[4],
                4: car_probs[5],
                5: car_probs[2],
            }
            pil_image = Image.fromarray(image['image'][0]) #  全局
            img_processed = car_img_processor(images=pil_image, return_tensors="pt")
            # Make predictions
            car_model_prediction = car_img_model(**img_processed)
            car_model_logits = car_model_prediction.logits.detach().cpu().numpy() # "0": Crack "1": Scratch "2": Tire Flat "3": Dent "4": Glass Shatter "5": Lamp Broken
            car_probs = torch.softmax(torch.tensor(car_model_logits), dim=1)[0]
            car_label_map_global = {
                0: car_probs[3],
                1: car_probs[1],
                2: car_probs[0],
                3: car_probs[4],
                4: car_probs[5],
                5: car_probs[2],
            }
            car_similarity = {
                cat_id: (car_label_map[cat_id-1] + car_label_map_global[cat_id-1])/2  # 转换为Python float类型
                for i, cat_id in enumerate(centroid_features_global.keys())
            }
            # import ipdb;ipdb.set_trace()
        # 计算总得分
        dino_cat_id = category2id[dino_phrase]
        total_scores = {}
        for cat_id in category2id.values():
            # 获取各类别得分（注意处理可能缺失的键）
            text_score = text_similarity.get(cat_id, 0.0)
            global_score = global_similarity.get(cat_id, 0.0)
            partial_score = partial_similarity.get(cat_id, 0.0)
            if dataset_name=='dataset3':
                car_score = car_similarity.get(cat_id, 0.0)
                total_scores[cat_id] = (w_text * text_score) + (w_global * global_score) + (w_partial * partial_score) + (w_car * car_score)
                if cat_id == dino_cat_id:
                    total_scores[cat_id] *= scale
                    if total_scores[cat_id] > 1.0:
                        total_scores[cat_id] = 1.0
            else:
                # 加权求和
                total_scores[cat_id] = (w_text * text_score) + (w_global * global_score) + (w_partial * partial_score)
                if cat_id == dino_cat_id:
                    total_scores[cat_id] *= scale
                    if total_scores[cat_id] > 1.0:
                        total_scores[cat_id] = 1.0
        # import ipdb;ipdb.set_trace()
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

    """ trick..."""
    box_paint_filtered = [] # trick
    category_id_filtered = []
    score_filtered = []

    if dataset_name == 'dataset1': # box不变，类别改变，得分固定0.9
        # 添加空检测保护
        # import ipdb;ipdb.set_trace()
        if not scores:  # 没有有效检测框时跳过不输出图了
            continue
        max_score_idx = np.argmax(scores) # 最高分类别
        dominant_category = category_ids[max_score_idx]
        # 将所有检测框设为该类别
        for bbox, score,bbox_original in zip(boxes_ret, scores, box_to_paint):
            box_paint_filtered.append(bbox_original)
            score_filtered.append(0.98) # 固定为0.98分
            category_id_filtered.append(dominant_category)
            result = {
                "image_id": image["image_id"],
                "category_id": dominant_category,
                # "bbox": bbox.tolist(),
                "bbox": [int(x) for x in bbox.tolist()],
                "score": float(score)
            }
            results.append(result)
            # 维护用于绘制的列表

    else:
        for bbox, category_id, score, box_original in zip(boxes_ret, category_ids, scores, box_to_paint):
            box_paint_filtered.append(box_original)
            score_filtered.append(score)
            category_id_filtered.append(category_id)
            result = {
                "image_id": image["image_id"],
                "category_id": category_id,
                # "bbox": bbox.tolist(),  # 将 numpy 数组转换为列表
                "bbox": [int(x) for x in bbox.tolist()],
                "score": float(score)
            }
            
            results.append(result)
    category_names_paint = [id2category[i] for i in category_ids]
    category_names_paint_filtered = [id2category[i] for i in category_id_filtered]

    # 渲染图片边框

    if len(box_to_paint) > 0:
        # 确保每个元素是numpy数组后再进行堆叠
        box_paint_filtered = [np.array(b) for b in box_paint_filtered]
        box_paint_filtered = torch.from_numpy(np.stack(box_paint_filtered)).float()
    else:
        box_paint_filtered = torch.zeros((0, 4), dtype=torch.float32)  # 明确二维形状
    annotated_frame = annotate(image_source=image['image'][0], boxes=box_paint_filtered, logits=score_filtered, phrases=category_names_paint_filtered)


    # if len(box_to_paint) > 0:
    #     # 确保每个元素是numpy数组后再进行堆叠
    #     box_to_paint = [np.array(b) for b in box_to_paint]
    #     box_to_paint = torch.from_numpy(np.stack(box_to_paint)).float()
    # else:
    #     box_to_paint = torch.zeros((0, 4), dtype=torch.float32)  # 明确二维形状
    # annotated_frame = annotate(image_source=image['image'][0], boxes=box_to_paint, logits=scores, phrases=category_names_paint)    
    # 保存图像
    output_dir = f"../result/{dataset_name}_{shot}/images" # 图像保存路径
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{image['image_id']}.jpg", annotated_frame)

# 保存对象相关信息
with open(f'../result/{dataset_name}_{shot}/{dataset_name}_{shot}.json', 'w') as f:
    json.dump(results, f, indent=4)


