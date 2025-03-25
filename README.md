# ProtoDINO

我们在Grounding_DINO_FineTuning仓库的基础上改进得到了当前代码。

## 权重下载

grounding_dino权重下载 

wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

CLIP实现采用了DFN5B-CLIP-ViT-H-14-378，下载模型请使用

`git lfs clone https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378`

car-damage-detector模型下载模型请运行

`git lfs clone https://huggingface.co/beingamit99/car_damage_detection`

## 运行代码
首先应准备数据集，并按图中安排目录结构。数据放在datasets下。

<img width="237" alt="image" src="https://github.com/user-attachments/assets/4da559a7-383a-450d-a193-1588ff7d8100" />

在ProtoDINO/configs下修改配置文件，

在`train_config`和`test_config`中应注意修改`weights_path`和`config_path`的路径

在`get_centroid_new.py` 和 `get_result_prototype_new.py`中，应确定CLIP和car-damage-detection的模型路径正确


我们已经生成了原型的pkl文件在项目中。想要重新构建原型，请运行`python get_centroid_new.py`, 

进行推理需要使用
`python get_result_prototype_new.py`

