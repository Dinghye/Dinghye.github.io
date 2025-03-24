---
title: 【Detectron2】Faster-RCNN：2.数据加载注册原理

tags:
  - Detectron2
categories:
  - - 机器学习
    - 原理
---

【深度研究Detectron2】，基于文档Digging into Detectron2。Detectron2 的数据注册原理。

<!--more -->

## 1. 网络中哪些地方使用了ground truth data？

为了训练检测模型，我们需要准备**图像和标注**
至于Base-RCNN-FPN(Faster R-CNN)，在区域建议网络(Region Proposal Network，RPN)和箱头(Box Head)中使用的是ground truth数据(见图1)。

> 在有监督学习中，数据是有标注的，以(x, t)的形式出现，其中x是输入数据，t是标注。**正确的t标注是ground truth，** 错误的标记则不是。（也有人将所有标注数据都叫做ground truth）

<img src="https://i.loli.net/2020/10/31/5AfveKbuG3sZQFq.png" alt="图1.ground-truth箱注释用于RPN和Box Head计算损失" style="zoom:50%;" />



用于物体检测的注释数据包括：

* **方框标签**：对象的位置和大小(如\[x，y，w，h\])
* **类别标签**：对象的类别id(如12：“parking meter”)

需要注意的是，[RPN](https://dinghye.gitee.io/2020/10/31/Detectron2FPN/)<u>不会学习对对象类别进行分类</u>，所以类别标签只在[ROI Head](https://dinghye.gitee.io/2020/11/01/Detectron2ROI/)处使用。从指定数据集的注释文件中加载标注数据。我们来看一下数据加载的过程。

## 2. 数据加载器

Detectron 2的数据加载器是多级嵌套的。它是在开始训练³之前由[构建器](https://github.com/facebookresearch/detectron2/blob/1a7daee064eeca2d7fddce4ba74b74183ba1d4a0/detectron2/data/build.py#L255-L385)构建的。

-   *dataset\_dicts (list)*是一个从数据集注册的注释数据的列表。
-   [DatasetFromList](https://github.com/facebookresearch/detectron2/blob/1a7daee064eeca2d7fddce4ba74b74183ba1d4a0/detectron2/data/common.py#L58-L81)(*data.Dataset*)取一个*dataset\_dicts*，并将其包装成一个torch数据集。
-   (*data.Dataset*)*调用*[DatasetMapper](https://github.com/facebookresearch/detectron2/blob/1a7daee064eeca2d7fddce4ba74b74183ba1d4a0/detectron2/data/dataset_mapper.py#L19-L147)类来映射DatasetFromList的每个元素。它加载图像，转换图像和注解，并将注解转换为\"Instances\"对象。

<img src="https://i.loli.net/2020/11/01/neBpQ6UWrdN14kz.png" alt="图3.Detectron 2的数据加载器Detectron 2的数据加载器" style="zoom: 67%;" />



## 3. 加载注解数据

假设我们有一个名为\'*mydataset*\'的数据集，它的图片和注释如下。

![图4.图像和注释的例子](https://i.loli.net/2020/11/01/gZuVN3owjnSDfzO.png)



要从一个数据集加载数据，**必须将它注册到DatasetCatalog**。例如，要注册*mydataset*。

从detectron2.data导入DatasetCatalog。

```python
from mydataset import load_mydataset_jsondef register_mydataset_instances(name, json_file):
(name, lambda: load_mydataset_json(json_file, name))
```

并调用 *register\_mydataset\_instances* 函数，指定你的 json 文件路径。

*load\_mydataset\_json*函数必须包含一个json加载器，这样才能返回下面的dict记录列表。

```json
[{
	'file_name': 'imagedata_1.jpg', # 图片文件名。
	'height': 640, # 图片高度
	'width': 640, # 图片宽度
	'image_id':12, # image id
	'annotations':[ # 注释列表
		{
    		'iscrowd':0, #人群标志
			'bbox':[180.58, 162.66, 24.20, 18.29], # 界线盒标签。
			'category_id': 9, # 类别标签
			'bbox_mode': < BoxMode.XYWH_ABS: 1>}}。 # 盒式坐标模式
			,...
		
]},
,...
]
```

对于<u>COCO数据集(Detectron 2的默认值)</u>，*load_coco_json*[函数](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py#L29-L194)起到了作用。

## 4. 映射数据

在训练过程中，注册的标注记录会被逐一挑选出来。我们需要实际的图像数据（不是路径）和相应的注释。数据集映射器[*DatasetMapper*](https://github.com/facebookresearch/detectron2/blob/1a7daee064eeca2d7fddce4ba74b74183ba1d4a0/detectron2/data/dataset_mapper.py#L19-L147)处理记录，为*数据集\_dict*添加\'图像\'和\'实例\'。\'Instances\'是Detectron 2的地真结构对象。

1.  加载和转换图像通过[read_image](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/detection_utils.py#L36-L70)函数
    加载由\"文件名\"指定的图像。加载的图像[通过预定义的变换器}](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/dataset_mapper.py#L78-L81)（如左右翻转）进行变换，最后注册形状为（通道、高度、宽度）的图像张量。

2.  转换标注数据集
    的\"注释\"是通过对图像进行转换而转换的。例如，如果图像已被翻转，则方框坐标将被改变为翻转的位置。

3.  将标注转换为实例（Instances）
    
    [这个函数](https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/detection_utils.py#L234-L257)在数据集映射器中调用，将标注
    转换为实例。bbox\'注解被注册到*Boxes*结构对象中，它可以存储一个边界盒的列表。\'category\_id\'注解被简单地转换为一个火炬张量。

映射后，*dataset\_dict*应该长成：

```json
{'file_name': 'imagedata_1.jpg',
 'height': 640, 
 'width': 640, 
 'image_id':0,
 'image': tensor([[[255., 255., 255., ..., 29., 34., 36.], ...[169., 163., 162., ..., 44., 44., 45，]]]),
 'instances':
     'gt_boxes':Boxes(tensor([[100.55, 180.24, 114.63, 103.01],[180.58, 162.66, 204.78, 180.95]])),
     'gt_classes': tensor([9, 9]),
}。
```

现在我们有了图像和地道的注释，Detectron 2模型可以进行学习啦！