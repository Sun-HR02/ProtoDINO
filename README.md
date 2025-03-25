# ProtoDINO: Cross-Domain Few-Shot Object Detection via GroundingDINO and CLIP-Based Prototypes

本仓库是 NTIRE 2025 CD-FSOD Challenge - ProtoDINO: Cross-Domain Few-Shot Object Detection via GroundingDINO and CLIP-Based Prototypes 的官方实现。

为了增强基准模型目标分类的能力，ProtoDINO基于开放集目标检测模型GroundingDINO, 引入CLIP模型在少量目标域样本上提取图像局部特征和全局特征作支持集，并分别构建局部原型和全局原型网络。在目标检测过程中，对每一个图像查询，我们使用CLIP提取视觉特征，计算其与局部原型和全局原型的L2距离，并把该距离作为目标分类的指标之一；此外，我们引入CLIP模型进行分类；我们还引入car-damage-detection 模型，这是一个基于ViT实现的车辆外观损伤分类模型。在目标分类中，我们将grounding_dino、CLIP、car-damage-detection以及原型网络匹配概率加权求和，作为分类指标。

我们在Grounding_DINO_FineTuning仓库的基础上改进得到了当前代码。

## 模型权重

grounding_dino权重下载 

`wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`

CLIP实现采用了DFN5B-CLIP-ViT-H-14-378，下载模型请使用

`git lfs clone https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378`

car-damage-detector模型下载模型请运行

`git lfs clone https://huggingface.co/beingamit99/car_damage_detection`

## 数据集
首先应准备数据集，并按图中安排目录结构。将COCO格式的数据放在datasets下。

<img width="237" alt="image" src="https://github.com/user-attachments/assets/4da559a7-383a-450d-a193-1588ff7d8100" />

在项目代码中，应确保权重和模型文件路径正确：

* ProtoDINO/configs下修改配置文件，在`train_config`和`test_config`中应注意修改`weights_path`和`config_path`的路径

* 在`get_centroid_new.py` 和 `get_result_prototype_new.py`中，应确定CLIP和car-damage-detection的模型路径正确

## 开始推理

我们已经生成了原型的pkl文件在项目中。想要重新构建原型，请运行`python get_centroid_new.py`, 

运行`python get_result_prototype_new.py`获取目标检测结果，生成结果会生成在`results`目录中


