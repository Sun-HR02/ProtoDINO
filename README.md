# ProtoDINO

我们在Grounding_DINO_FineTuning仓库的基础上改进得到了当前代码。

## 权重下载

grounding_dino权重下载 wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

CLIP实现采用了DFN5B-CLIP-ViT-H-14-378，使用`git lfs clone https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378`下载模型

car-damage-detector模型请通过`git lfs clone https://huggingface.co/beingamit99/car_damage_detection`下载模型

## 运行代码
首先应准备数据集，并按图中安排目录结构。数据放在datasets下。
<img width="237" alt="image" src="https://github.com/user-attachments/assets/4da559a7-383a-450d-a193-1588ff7d8100" />
在ProtoDINO

构造原型需要使用
`python get_centroid_new.py`

进行推理需要使用
`python get_result_prototype_new.py`

具体使用文档还在编写当中....
