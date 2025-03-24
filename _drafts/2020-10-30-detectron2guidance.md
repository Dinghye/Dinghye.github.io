---
title: 【Detectron2】Detectron2 用自己的数据训练使用Faster-RCNN

tags:
- Detectron2
categories:
- - 机器学习	
  - 实践操作
---

detectron2 入门笔记，数据格式转换，模型训练、检测、评价

<!--more -->

参考：https://www.cnblogs.com/rioka/p/13781852.html

# 1. 项目目录

```shell
Project
--COCO-Detection
    faster_rcnn_R_50_FPN_3x.yaml  # 官方代码库 copy
    Base-RCNN-FPN.yaml  # 同上
    my_config.yaml  # 自己的训练配置文件
--tools
    train_net.py  # 官方代码库 copy 并自行修改
    predictor.py  # 模型预测 输出
    single_predictor.py # 单张图片模型输出
    train.sh
    train_resume.sh
    eval.sh
--utils
    csv2coco.py  # 将自己的数据集转化为标准 coco 数据集
    cocoDatasetViewer.py # coco数据集展示
    dataReview.py  # 训练数据展示

```

数据集目录为

```shell
--MyDataset   # 数据集
----annotations
	  train.json
	  val.json
----images
----train
      1.jpg
      ....
----val
      2.jpg
      ....
```

# 2. 数据格式转换

## 2.1 COCO数据格式

COCO的 全称是Common Objects in COntext，是微软团队提供的一个可以用来进行图像识别的数据集。MS COCO数据集中的图像分为训练、验证和测试集。

COCO数据集现在有3种标注类型：**object instances（目标实例）, object keypoints（目标上的关键点）, 和image captions（看图说话）**，使用JSON文件存储。

这里我们使用的是instances（框住目标），用于进行目标检测。一些仓库提供了对应的数据转换程序

> 参考：https://github.com/spytensor/prepare_detection_dataset

![目录](https://i.loli.net/2020/11/01/6Ctiljarxv7UAPy.png)

## 2.2 当前数据格式与转换

### 2.2.1 数据格式

根据对应说明有：训练集中每张图像数据对应一个 txt 标注文件,如图像数据名称为 1.tif,则对应的标注文件为 1.txt, 标注文件每一行为 cls x1 y1x2 y2 x3 y3 x4 y4 共 5 个参数,间隔字符为空格,标注对象的个数
对应存储的行数。其中(x1,y1),(x2,y2),(x3,y3),(x4,y4)四个点构成对象的标注框。同时，图像数据均为1024x1024的tif格式文件。

### 2.2.2 正框转换

然而值得注意的是，这里的标注是**斜框**，而一般来说的coco程序转换的都是正框。所以这里对原数据进行了获取正框的处理，**但是实际上这样操作会导致模型精度降低**。

```Python
 # 这里进行数据格式的转换，原来csv代码使用的是path,xmin, ymin, xmax, ymax,label的格式
        # specially , we used absolute path
        formalnum = [path + "images/" + str(name) + ".jpg", min(num[1], num[3], num[5], num[7]),
                     min(num[2], num[4], num[6], num[8]),
                     max(num[1], num[3], num[5], num[7]), max(num[2], num[4], num[6], num[8]), str(int(num[0]))]
```

通过 [Github](https://github.com/spytensor/prepare_detection_dataset) 这样的转换，将现有数据转换为前面csvTococo的预数据。得到对应的coco数据集

> ![数据集路径](https://i.loli.net/2020/11/01/Av9WBQpDURwFZhN.png)

## 2.3 COCO数据预览

在转换完数据之后，理论上要对转换后的coco数据进行一次预览，检查是否已经转换好了（然而我最开始的时候并没有做这一步，训练完模型之后发现结果很差，再反过头来发现自己有个地方写错了导致要重新来过）。

选择链接好对应的annotations和images

```Python
from pycocotools.coco import COCO
import cv2 as cv
import os
import numpy as np
import random

img_path = '../MyDataset/images/val'  # 把图片直接放在同一文件夹下
annFile = '../MyDataset/annotations/val.json'  # 同样

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())

catIds = coco.getCatIds(catNms=['1', '2', '3', '4', '5'])
imgIds = coco.getImgIds(catIds=catIds)
img_list = os.listdir(img_path)
for i in range(len(img_list)):
    img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
    image_name = img['file_name']

    # 加了catIds就是只加载目标类别的anno，不加就是图像中所有的类别anno
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    anns = coco.loadAnns(annIds)

    coco.showAnns(anns)

    coordinates = []
    img_raw = cv.imread(os.path.join(img_path, image_name))
    for j in range(len(anns)):
        x1 = int(anns[j]['bbox'][0])
        y1 = int(anns[j]['bbox'][1] + anns[j]['bbox'][3])
        x2 = int(anns[j]['bbox'][0] + anns[j]['bbox'][2])
        y2 = int(anns[j]['bbox'][1])
        cv.rectangle(img_raw,
                     (x1, y1),
                     (x2, y2),
                     (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                     2)
    cv.imshow('image', img_raw)
    cv.waitKey()

```

预览结果如下，不同框的颜色就是不同的分类：

![ground-truth box](https://i.loli.net/2020/11/01/dRZbeSyB7u9j2cl.png)

# 3. 模型训练

## 3.1 数据注册

使用 CocoFormat 的数据集是最优雅的做法，也可以按照官方给定的方法在 `tools/rain_net.py` 中自定义数据集。

这里使用前面已经转换好的coco数据集，并将数据集初测代码包装成类，放在tools/train_net.py中

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

class Register:
    """用于注册自己的数据集"""
    CLASS_NAMES = ['__background__', '1', '2', '3', '4', '5']  # 保留 background 类
    ROOT = "../MyDataset"

    def __init__(self):
        self.CLASS_NAMES = Register.CLASS_NAMES or ['__background__', ]
        # 数据集路径
        self.DATASET_ROOT = Register.ROOT
        # ANN_ROOT = os.path.join(self.DATASET_ROOT, 'COCOformat')
        self.ANN_ROOT = self.DATASET_ROOT

        self.TRAIN_PATH = os.path.join(self.DATASET_ROOT, 'images/train')
        self.VAL_PATH = os.path.join(self.DATASET_ROOT, 'images/val')
        self.TRAIN_JSON = os.path.join(self.ANN_ROOT, 'annotations/train.json')
        self.VAL_JSON = os.path.join(self.ANN_ROOT, 'annotations/val.json')
        # VAL_JSON = os.path.join(self.ANN_ROOT, 'test.json')

        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            "coco_my_train": (self.TRAIN_PATH, self.TRAIN_JSON),
            "coco_my_val": (self.VAL_PATH, self.VAL_JSON),
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (image_root, json_file) in self.PREDEFINED_SPLITS_DATASET.items():
            self.register_dataset_instances(name=key,
                                            json_file=json_file,
                                            image_root=image_root)

    @staticmethod
    def register_dataset_instances(self, name, json_file, image_root):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """

        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        MetadataCatalog.get(name).set(json_file=json_file,
                                      image_root=image_root,
                                      evaluator_type="coco")

    def plain_register_dataset(self):
        """注册数据集和元数据"""
        # 训练集
        DatasetCatalog.register("coco_my_train", lambda: load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH))
        MetadataCatalog.get("coco_my_train").set(thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                                 evaluator_type='coco',  # 指定评估方式
                                                 json_file=self.TRAIN_JSON,
                                                 image_root=self.TRAIN_PATH)
        # DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
        # 验证/测试集
        DatasetCatalog.register("coco_my_val", lambda: load_coco_json(self.VAL_JSON, self.VAL_PATH))
        MetadataCatalog.get("coco_my_val").set(thing_classes=self.CLASS_NAMES,  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                               evaluator_type='coco',  # 指定评估方式
                                               json_file=self.VAL_JSON,
                                               image_root=self.VAL_PATH)

    def checkout_dataset_annotation(self, name="coco_my_val"):
        """
        查看数据集标注，可视化检查数据集标注是否正确，
        这个也可以自己写脚本判断，其实就是判断标注框是否超越图像边界
        可选择使用此方法
        """
        # dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
        dataset_dicts = load_coco_json(self.TRAIN_JSON, self.TRAIN_PATH)
        print(len(dataset_dicts))
        for i, d in enumerate(dataset_dicts, 0):
            # print(d)
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
            vis = visualizer.draw_dataset_dict(d)
            # cv2.imshow('show', vis.get_image()[:, :, ::-1])
            cv2.imwrite('out/' + str(i) + '.jpg', vis.get_image()[:, :, ::-1])
            # cv2.waitKey(0)
            if i == 200:
                break
```

同时在main函数中调用

```Python
def main(args):
    cfg = setup(args)
    Register().register_dataset()  # register my dataset
    ...
```

## 3.2 编辑配置文件

为了更加清晰，选择不更改原始的依赖配置，而是新建一个my_config.yaml

```yaml
_BASE_: "faster_rcnn_R_50_FPN_3x.yaml"
DATASETS:
  TRAIN: ("coco_my_train",)
  TEST: ("coco_my_val",)
MODEL:
  RETINANET:
    NUM_CLASSES: 6  # 类别数+1, 因为有background
  # WEIGHTS: "../tools/output/model_final.pth"
SOLVER:
  # IMS_PER_BATCH: 16
  # 初始学习率
  BASE_LR: 0.00025
  # 迭代到指定次数，学习率进行衰减
  # STEPS: (210000, 250000)
  # MAX_ITER: 270000
  CHECKPOINT_PERIOD: 1000
TEST:
  EVAL_PERIOD: 3000
```

## 3.3 模型训练

 为了方便进行模型训练，将对应的参数写好成sh，即可进行训练操作

```shell
################ train.sh ################
# Linux 下换行符为 CRLF 的需改为 LF
# lr = 0.00025 * num_gpus
python3 train_net.py \
  --config-file ../configs/my_config.yaml \
  --num-gpus 1 \
  SOLVER.IMS_PER_BATCH 4 \
  SOLVER.BASE_LR 0.01 \
  SOLVER.MAX_ITER 3000 \
  SOLVER.STEPS '(2400, 2900)'

############# train_resume.sh #############
# 断点续 train
# --num-gpus 不能省略
python3 train_net.py \
  --config-file ../configs/my_config.yaml \
  --num-gpus 1 \
  --resume

################# eval.sh #################
python3 train_net.py \
  --config-file ../configs/my_config.yaml \
  --eval-only \
  MODEL.WEIGHTS output/model_final.pth
```

值得注意的是，进行训练的时候，很容易出现cuda run out of memory错误。（参考Linux进行解决）同时建议在使用的时候`watch nvidia-smi`来检测cuda的使用，及时kill掉没有用的进程。

开始训练：

![开始训练](https://i.loli.net/2020/11/01/i9jJTlcFAoE7ba1.png)

![数据统计结果](https://i.loli.net/2020/11/02/1y9b4w8ixHaEZNC.png)

训练结果：输出output

![模型](https://i.loli.net/2020/11/01/ctHJlwhfuirUEo1.png)

# 4. 模型使用

## 4.1 模型评价

使用eval.sh对跑出来的模型进行评价：

![模型评价](https://i.loli.net/2020/11/01/936KvpJzmElakSe.png)

![模型评价2](https://i.loli.net/2020/11/01/EHrbt4LeJK6ynv9.png)

发现结果非常不好，原因有一下几点：

1. 参数是随便输入的，batch数目也比较小，需要进行调整
2. 原始标注数据是斜框的，为了转成coco，转变成了正框，使得精度有一定的损失

## 4.2 检测新图片

这里参照原先跑demo用的代码进行某张图片的检测

```python
import os
import cv2
from detectron2.utils.logger import setup_logger

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def detect(im, name):
    outputs = predictor(im)
    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes

    # 在原图上画出检测结果
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(name.split('.')[0] + '.jpg', v.get_image()[:, :, ::-1])


if __name__ == "__main__":
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file("output/config.yaml")
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 模型阈值
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    predictor = DefaultPredictor(cfg)
    img = cv2.imread("../MyDataset/test/6.jpg")
    detect(img, "6.jpg")
```

特别值得注意的是，在使用的时候一开始出现了颜色通道错误的问题（左边为通道错误，检测结果，右边为原图），需要调整通道来进行输出。[detectron和颜色通道]()

![颜色通道](https://i.loli.net/2020/11/01/dlAcoG2xRWVqEwZ.png)

结果为：

<img src="https://i.loli.net/2020/10/30/d2GgpQFt98WeRUK.png" alt="image-20201030151243461" style="zoom:80%;" />

# 5. 总结

目前基本的使用操作已经基本掌握，但是效果还是不是很好，还有以下几点需要学习提高

1. 各种超参数的含义与使用
2. coco数据格式斜框处理办法