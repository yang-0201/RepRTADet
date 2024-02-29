# Rep-RTADet
This is the official code repository for "Rep-RTADet: Reparameterized Real-Time Algae Object Detectors Enhanced through Dynamic Cache-Based Poisson Fusion". 


`Rep-RTADet` 算法在阿里天池 [IEEE Cybermatics 第二届国际 "Vision Meets Algae" 挑战赛](https://tianchi.aliyun.com/competition/entrance/532171)中获得冠军

### 环境配置

```shell
conda create -n mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate mmyolo
pip install openmim
mim install "mmengine>=0.6.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0,<4.0.0"
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo
# Install albumentations
pip install -r requirements/albu.txt
# Install MMYOLO
mim install -v -e .
```

### 数据集配置
以 reprtadet_l_possion.py 配置文件为例
```
data_root  数据根目录
train_ann_file 训练集标注文件路径（json格式）
train_data_prefix 训练集图片路径
val_ann_file 验证集标注文件路径（json格式）
val_data_prefix 验证集图片路径
test_image_info 测试集标注文件路径（json格式）
test_image 测试集图片路径
```
### 数据集文件结构
```
├── algae
│   ├── images
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   ├── annotations
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   ├── instances_test.json
```
### 训练命令
``` shell
python tools/train.py configs/reprtadet/reprtadet_m.py
```
### 推理命令
``` shell
# val
python tools/test.py configs/reprtadet/reprtadet_m.py RepRTADet-m.pth
```
``` shell
# test
python tools/test.py configs/reprtadet/reprtadet_m_test.py RepRTADet-m.pth
```
### 模型和结果

 Model  | img size | box AP0.5 val | box AP val | box AP test | TTA  box AP test | 预训练模型                                                                                                                                                              | epochs
 ---- |----------|---------------|------------|------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------
  [RepRTADet-m](https://github.com/yang-0201/RepRTADet/releases/download/v1.0.0/RepRTADet-m.pth)  | 1280    | 0.934         | 0.723      |            | 0.7515           | [RTMDet-m](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth) | 200
  [RepRTADet-m2](https://github.com/yang-0201/RepRTADet/releases/download/v1.0.0/RepRTADet-m2.pth)  | 1280    | 0.933         | 0.722      | 0.7460     | 0.7510           | [RTMDet-m](https://download.openmmlab.com/mmyolo/v0/rtmdet/rtmdet_m_syncbn_fast_8xb32-300e_coco/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth) | 200

### Acknowledgements
* MMYOLO [https://github.com/open-mmlab/mmyolo/tree/main)
