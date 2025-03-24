---
title: 【Detectron2】Rotated Faster RCNN 利用Detectron2 训练自己的数据集
tags:
  - Detectron2
categories:
  - - 机器学习
    - 实践操作

---


本文将提到：1. 如何利用Detectron2 定义训练自己格式的数据集 2. Faster RCNN相关知识 3. Rotated Faster RCNN 的实现 4. 相关参数的设置

<!-- more -->

# 0. 说明&先修知识

在前面的已经介绍过关于如何使用coco数据集来使用Detectron2进行训练[Detectron2 用自己的数据训练使用Faster-RCNN](https://dinghye.gitee.io/2020/10/30/detectron2guidance/)

同时也学习了关于FasterRCNN的结构知识[Detectron2& Faster-RCNN指北](https://dinghye.gitee.io/2020/11/02/Detectron2Index/)

但是在实际使用的时候我们发现简单的这样训练效果并不是很好，原因有：

1. 本来是斜框数据，为了转成coco训练转成了正框，造成了精度损失
2. 参数没有进行调整

故本文主要用以介绍，如何使用Detectron2进行斜框数据的训练以及FasterRCNN的相关参数调整

本文项目目录结构如下：

```shell
Project
--COCO-Detection
    faster_rcnn_R_50_FPN_3x.yaml  # 官方代码库 copy
    Base-RCNN-FPN.yaml  # 同上
    my_config.yaml  # 自己的训练配置文件
--tools
	data_loader.py 
    train_net.py  # 官方代码库 copy 并自行修改
    predictor.py  # 模型预测
    train.sh
    train_resume.sh
    eval.sh
--utils
    dataReview.py  # 训练数据展示
```

***参考：***

[官方文档 Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html)

[官方文档：Detectron2 Tutorial.ipynb](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0)

[Example for rotated Faster RCNN](https://github.com/facebookresearch/detectron2/issues/21#issuecomment-607951599)

[使用Detectron2分6步进行目标检测](https://www.cnblogs.com/panchuangai/p/13875644.html)

# 1. 自定义数据注册

## 1.1 所需数据内容

Detectron2 中定义有标准数据集字典，这些字段能够帮助进行检测，分割等任务

| Task                            | Field                           |
| ------------------------------- | ------------------------------- |
| Common                          | file_name,height,width,image_id |
| Instance detection/segmentation | annotations                     |
| Semantic segmentation           | sem_seg_file_name               |
| Panoptic segmentation           | pan_seg_file_name,segments_info |

详细的字典定义解释参见[官方文档](https://detectron2.readthedocs.io/tutorials/datasets.html)。以FasterRCNN为例，其训练时所需要的主要用到的数据有：

```python
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

bboxMode，可以在detectron2.structure.boxes中看到，一共是有四种类型的box（XYXY_ABS、XYWH_ABS、XYXY_REL(not supported yet)、XYXY_REL(not supported yet)、XYWHA_ABS），每一种对应不同的数据格式。

## 1.2 斜框（XYWHA_ABS）BBOX计算

可以看到，如果我们需要进行斜框的数据集训练的话，那就需要使用**XYWHA_ABS**的数据格式。其bbox的格式即为[cx,cy,w,h,a]，其中cx，cy，为中心坐标，a为旋转角度（角度制）。所以我们需要将目前四个点格式转换成旋转角度的坐标，大概是这个样子：

![旋转计算示意图](https://i.loli.net/2020/11/10/nzXHmKLasj7guwb.png)

其中的数学转换可以参见[How to calculate rotation angle from rectangle points?](https://stackoverflow.com/questions/13002979/how-to-calculate-rotation-angle-from-rectangle-points) 值得注意的是：

* 需要使用atan2（而不是atan）函数来计算，以免出现角度计算不出来的问题
* 需要转换成角度，而不是弧度制

```python
def get_abbox(self, annotation):
    """
    purpose: 用于返回对应的bbox数据
    :param annotation: 输入的数组，其格式为：[label,x1,y1,x2,y2,x3,y3,x4,y4]
    :return: 输出XYWHA_ABS格式的bbox，其格式为：[centerX, centerY, w, h, a]（a为旋转角度）
    """
    centerx = (annotation[1] + annotation[3] + annotation[5] + annotation[7]) / 4
    centery = (annotation[2] + annotation[4] + annotation[6] + annotation[8]) / 4
    h = math.sqrt(math.pow((annotation[1] - annotation[3]), 2) + math.pow(
        (annotation[2] - annotation[4]), 2))
    w = math.sqrt(math.pow((annotation[1] - annotation[7]), 2) + math.pow(
        (annotation[2] - annotation[8]), 2))
    a = - math.degrees(math.atan2((annotation[8] - annotation[2]), (annotation[7] - annotation[1])))
    return [centerx, centery, w, h, a]
```



## 1.3 斜框数据注册

在数据注册的时候，我们使用到的是`DatasetCatalog.register(name:str,func:Any)`以及`MetadataCatalog.get(name:str).set(**kwargs:Any)`。

其中的name即为我们给自己数据名称取的名字，而func要求我们对数据进行一个返回处理。于是我们设置一个`get_dict`函数来定义我们自己的数据集，如下：

```python
 def get_dicts(self, ids):
    """
    purpose: 用于定义自己的数据集格式，返回指定对象的数据
    :param ids: 需要返回数据的名称（id）号
    :return: 指定格式的数据字典
    """
    dataset_dicts = []
    data = {}
    count = 0
    for i in ids:
        count += 1
        # 构建图！
        img = self.read_labels(self.DATA_PATH, i)   # read_label为读取txt的函数，返回的是关于这张图的标注数组
        data["ids"] = count
        data["image_id"] = int(i)
        data["height"] = 1024
        data["width"] = 1024
        data["file_name"] = self.DATA_PATH + "images/" + str(i) + ".tif"
        # 对于每一个图里面的annotation来说
        annotations = []
        for j in img:
            ann = {}
            ann["bbox_mode"] = BoxMode.XYWHA_ABS
            ann["category_id"] = int(j[0])  # 根据给出来的数据格式，0
            ann["bbox"] = self.get_abbox(j)
            annotations.append(ann)
            ann = {}
        data["annotations"] = annotations
        dataset_dicts.append(data)
        data = {}
    return dataset_dicts
```

这时，我们就可以把所有的数据都读入注册。但是在训练的过程中，我们往往还需要进行训练集和验证集的划分，这里我们参考前面csv2coco使用的，sklearn包中的`train_test_split`函数，这个函数可以随机按照比例来进行训练验证集的划分。我们这里选用0.2的比例：

```python
def register_dataset(self):
    """
    purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
    注册数据集（这一步就是将自定义数据集注册进Detectron2）
    """
    # 这里利用train_test_split函数（sklearn包）进行测试集和训练集的划分
    total_csv_annotations = range(1, 2009)
    total_keys = list(total_csv_annotations)
    train_keys, val_keys = train_test_split(total_keys, test_size=0.2)
    print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    
    # 注册进去数据
    self.plain_register_dataset(train_keys, val_keys)
```

`plain_register_dataset`用以直接这样来注册数据

```python
def plain_register_dataset(self, train_key, val_key):
    """注册数据集和元数据"""
    # 训练集
    DatasetCatalog.register("ship_train", lambda: self.get_dicts(train_key))
    MetadataCatalog.get("ship_train").set(thing_classes=self.CLASS_NAMES)  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
    
    # 验证/测试集
    DatasetCatalog.register("ship_val", lambda: self.get_dicts(val_key))
    MetadataCatalog.get("ship_val").set(thing_classes=self.CLASS_NAMES)  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
```

> 所有以上的函数内容均放在了class Register中，与前面的方式一样，我们只需要在train_net的main中第二行加入`Register().register_dataset()`就可以了。

## 1.4 斜框数据预览

注册完数据之后，我们希望能对斜框数据进行一个预览检查，看是否正确。斜框数据不同于正框数据，我们需要Visualizer 类进行继承重写一下：

```python
class myVisualization(Visualizer):
    """用于显示旋转过后的数据（继承Visualizer）"""
    def draw_dataset_dict(self, dic):
        annos = dic.get("annotations", None)
        if annos:
            if "segmentation" in annos[0]:
                masks = [x["segmentation"] for x in annos]
            else:
                masks = None
            if "keypoints" in annos[0]:
                keypts = [x["keypoints"] for x in annos]
                keypts = np.array(keypts).reshape(len(annos), -1, 3)
            else:
                keypts = None

            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWHA_ABS) for x in annos]

            labels = [x["category_id"] for x in annos]
            names = self.metadata.get("thing_classes", None)
            if names:
                labels = [names[i] for i in labels]
            labels = [
                "{}".format(i) + ("|crowd" if a.get("iscrowd", 0) else "")
                for i, a in zip(labels, annos)
            ]
            self.overlay_instances(labels=labels, boxes=boxes, masks=masks, keypoints=keypts)

        sem_seg = dic.get("sem_seg", None)
        if sem_seg is None and "sem_seg_file_name" in dic:
            sem_seg = cv2.imread(dic["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        if sem_seg is not None:
            self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
        return self.output
```

再进行一下测试预览：

```python
dataset_dicts = get_dicts(range(1,11)) # 获取编号1-10的图片信息
for d in dataset_dicts:
	e = dict(d)
	name = e.get("file_name")
	print(name)
	print(e.items())
	img = cv2.imread(name)
    visualizer = myVisualizer(img[:,:,::-1],metadata={},scale=1)
    vis = visualizer.draw_dataset_dic(e,)
    cv2.imshow("hi",vis.get_image()[:,:,::-1])
    cv2.waitkey(0)
```

![成功标注](https://i.loli.net/2020/11/10/mvD3egFkz9w51In.png)

可以看到标注是成功的，但是偶尔也有一些奇怪的东西（下图红色部分）：

<img src="https://i.loli.net/2020/11/10/2bRyVp64ANstcXk.png" alt="标注" style="zoom:50%;" />

就很迷惑，出现概率较低。一开始以为是公式前面有问腿或者怎么的，但是无论怎么调整都非常奇怪。所以怀疑有一部分的数据可能本身就存在问题。这一种情况可能需要进行讨论。

---

至此，即完成自己数据集的注册。在这里已经可以进行模型的训练和测试了（保留基础设置）。预测结果如下：

![检测结果1](https://i.loli.net/2020/11/10/PO438hme9sVyroH.png)

![检测结果2](https://i.loli.net/2020/11/10/E1fwedqFJD9OBc3.png)

我们可以看到，效果要比上一次正框的要好一些

<img src="https://i.loli.net/2020/10/30/d2GgpQFt98WeRUK.png" alt="正框训练结果，能检测但是比较乱" style="zoom: 80%;" />

> 同时，还有一个疑问，即为什么**输入的是斜框**，而输出预测的**结果却是正框**呢？

这个问题其实很简单，在前面的[核心Region Proposal Network](https://dinghye.gitee.io/2020/11/01/Detectron2DataLoader/)我们就学习到，FasterRCNN，实际上是生成对应形状的单元锚，和我们的ground truth data 进行计算loU来指导模型训练。而我们知道，在原来基本模型中，我们生成的是一票大大小小形状各异的**正框**。因此，我们的结果也是正框。

> 那为什么效果有提升了呢？在哪里有提升了？

在计算loU的时候，原本正框中有一部分实际上是不属于船身的，这一部分不属于船身的东西进入计算loU时，被当作船来算。这个过程产生了误差。而斜框数据保证了，标注数据内部都是船，从而使得loU计算更加精确，造成最后结果的提高。

# 3. 参数设置

那我们是否能够进行斜框训练，斜框的检测呢，也是有办法的。只不过需要对原始的模型进行微调整。github 上@st7ma784 等讨论关于Rotated FasterRCNN的训练以及参数配置[Example for rotated faster rcnn](https://github.com/facebookresearch/detectron2/issues/21)

## 3.1 设置对应参数

在原先my_config.yml上，我们添加更多关于模型的设置。包括将anchor生成器换成了`RotatedAnchorGenerator`，并给出了其生成的几个角度（30°为差），ROI_HEADS换成`RROIHeads`等。

```yaml
 Model:
  RPN:
    HEAD_NAME: "StandardRPNHead"
    BBOX_REG_WEIGHTS: (10,10,5,5,1)
  ANCHOR_GENERATOR:
    NAME: "RotatedAnchorGenerator"
    ANGLES: [ [ -60,-30,0,30,60,90 ] ]
  ROI_HEADS:
    NAME: "RROIHeads"
    BATCH_SIZE_PER_IMAGE: 256   # faster, and good enough for this toy dataset (default: 512)
    NUM_CLASSES: 6
    SCORE_THRESH_TEST: 0.5  # set threshold for this model
  PROPOSAL_GENERATOR:
    NAME: "RRPN"
  ROI_BOX_HEAD:
    POOLER_TYPE: "ROIAlignRotated"
    BBOX_REG_WEIGHTS: (10.0, 10.0, 5.0, 5.0, 1.0)
```

当然，不要忘记把Train和Test的名称修改（根据前面注册的数据）

```yaml
DATASETS:
  TRAIN: ("ship_train",)
  TEST: ("ship_val",)
```



# 4. 开始训练

## 4.1 训练过程

对部分mapper进行重写修改

```python
def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([T.Resize((800, 800))], image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        my_transform_instance_annotations(obj, transforms, image.shape[:2])
        ############################################
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=mapper)


def my_transform_instance_annotations(annotation, transforms, image_size, *, keypoint_hflip_indices=None):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYWHA_ABS  #######change back to xyxy

    return annotation
```



## 4.2 训练结果

最开始的时候训练使用的`BATCH_SIZE_PER_IMAGE`是512，以为会有更好的效果，但是实际使用的时候发现非常难收敛，total_loss训练了大概一天多之后，total_loss一直在0.4~0.8之间盘旋。停止训练测试当前模型效果如下：

![检测结果1](https://i.loli.net/2020/11/10/ztN5jxQ2H3rYcAa.png)

可以看到，的确检测框开始旋转了，但是它的效果并没有直接用正框效果好，出现了很多重叠检测的东西：

![检测结果2](https://i.loli.net/2020/11/10/rzf2YImKTMbEVHa.png)

![检测结果3](https://i.loli.net/2020/11/10/LxKsWOS3Q8IN9VM.png)

有的结果都看上去有些鬼畜了。但是仔细看这些结果容易发现，这个略带一些偏的anchor非常像我们前面数据中还存在的一些莫名其妙的地方比如：

![标记数据显示](https://i.loli.net/2020/11/10/tEldm7uUJnGeaKP.png)

原始数据中就有一些框是微微倾斜的（并且没有办法矫正）。所以结果是这样可能与这个有很大的关系。介于此，尝试直接读取原标记数据看看问题出在哪里：（cv2，polyline函数绘出）

![原始标记数据,明显非矩形](https://i.loli.net/2020/11/10/E6KlGodIHaSMcVA.png)

可以看到！原来的标记数据就不是一个方正的矩形，而是一个polyline。而在训练当中我们转换角度的时候，出现这样的角度微倾斜是有道理的（这也意味着两边长和宽都不是相等的）。

这个时候我们需要解决的不仅仅是旋转问题，非矩形也需要加入考虑。（这一篇先写到这里！）