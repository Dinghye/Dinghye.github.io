---
title: 【RS-Detection】RS-Detection-ToolKit-ReadMe

tags:
- 技巧
categories:
- - 机器学习
  - 实践操作
---

一个基于Detectron2 和pytorch的斜框目标检测的模板

<!--more-->

# RS-Detection-ToolKit 功能说明

> github：https://github.com/Dinghye/RS-Detection-ToolKit

# 0. Introduction

> 本模板主要是针对斜框目标检测，整理一套可以拿来就用的方法。

## 0.1 总体结构

本模板的主体逻辑结构如下图所示。

* 设计data_info的父类，子类json、txt等数据格式继承该类。可以使用DATA_INFO直接注册训练，也可也通过DATA_INFO内部方法：`self.info_to_rotated_coco`转换成coco数据格式，使用coco_Register()注册。coco格式数据集读写速度更快，能够一定程度上缩短训练时间。
* 设计my_Visualization,能够对斜框标注数据进行预览查看。`load_data_viewer`即对注册的数据集进行查看，保证输入结果的正确。`origin_data_viewer`即对原始数据集进行查看，查看数据标注的规范性等。（参见：https://dinghye.gitee.io/2020/11/10/detectron2guidance2/）
* 设计image_process：参考感谢：https://github.com/CAPTAIN-WHU/DOTA_devkit。详见1.
* 设计data_statistic：详见2. 

<img src="https://i.loli.net/2021/08/02/6xKf5XDgHN1hYwW.png" alt="architecture (修改中)" style="zoom: 50%;" />

## 0.2 数据接口

```json
data={
    "file_name": filename,  # instance的位置，建议使用绝对路径
    "image_id": idx,        # 对于每一个instance须有有一个唯一id
    "height": height,		# image的高度
    "width": width,			# image的宽度
    "annotations":[			# 对于这一张图片的标注,可以是多个
        {
            "bbox": [cx,cy,w,h,a]	# 和bbox_mode相关联
        	"bbox_mode": BoxMode.	
        	"category_id": 
        }
        {
            "bbox": [cx,cy,w,h,a]
        	"bbox_mode": BoxMode.
        	"category_id": 
        }
    ],
}
```



# 1. 数据裁剪（utils.ImageSplit）

> 本功能参考：https://github.com/CAPTAIN-WHU/DOTA_devkit

* Motivation：

  * 遥感单张数据尺寸较大，检测目标较小；这样的数据直接训练速度慢、难度高；
  * 遥感影像标注点的顺序（Sequential label points）关系到模型回归（详见【RS Detection】目标检测模型的选择（一个简单的overview）1.2.1（3））

* 处理方法：裁剪+标注点排序+规范名命：

  ```python
  1. split image by given size
  2. choose best point order to help regression (optional)
  3. image who was split will be renamed by "originname__1__left x__left y". 
  ```

* API(utils/ImageSplit.py)：`splitbase(json_info,outcome_dir)`

* example：

  <img src="https://i.loli.net/2021/08/02/1awcJUbfhZKy5Bn.png" alt="待裁剪前" style="zoom: 50%;" />

  

<img src="https://i.loli.net/2021/08/02/TwhlbYpe4LKPjfa.png" alt="裁剪后1" style="zoom:25%;" /><img src="https://i.loli.net/2021/08/02/7dhD9yWwXPSjuIs.png" alt="裁剪后2" style="zoom:25%;" />

> 裁剪之后部分框被damaged，针对这一类框，我们添加difficulty参数，当difficulty=2时，即目标已经被严重裁剪坏，方便之后统一处理。

# 2.  数据统计

* Motivation：通过对目标数据的统计，更好设计超参数和结构

* API(utils/data_statistic.py)：

  ```PYTHON
  1. the amount of data contained in each graph
  2. the aspect ratio distribution of the overall target
  3. the area distribution of the overall target
  4. bibliographic statistics of different categories of targets
  5. distribution of aspect ratios within a single category
  6. area distribution within a single category
  （NEW）7. 旋转角度统计(info type only)
  ```

* example：旋转角度统计结果（30°，60°）

  <img src="https://i.loli.net/2021/08/02/7WsVo5erB6lEchd.png" alt="angle_30" style="zoom: 49%;" /><img src="https://i.loli.net/2021/08/02/kFz7X6JATI5rCEO.png" alt="angle_60" style="zoom:50%;" />

# 3. 数据预览

* Motivation：检查输入数据正确性，数据标注特点

* API(utils/load_data_viewer.py utils/origin_data_viewer.py)

* example：origin_data_viewer能检查标注点的顺序，load_data_viewer能够看到对象的类别信息

  <img src="https://i.loli.net/2021/08/02/Po2kzAlcBdJaDTW.png" alt="origin_data_viewer" style="zoom:33%;" /><img src="https://i.loli.net/2021/08/02/1fnWcUaYjzMuP9s.png" alt="load_data_viewere" style="zoom: 33%;" />
