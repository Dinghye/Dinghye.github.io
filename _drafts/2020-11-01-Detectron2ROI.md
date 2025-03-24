---
title: 【深度学习】Faster-RCNN：4.ROI(Box) Head

tags:
  - Detectron2
categories:
  - - 机器学习
    - 原理
---

【深度研究Detectron2】，基于文档Digging into Detectron2。本篇主要讲到ROI (Box) Head。最后一部分！

<!-- more -->

本篇我们要深入最后一部分\--ROI（Box）Head³（见图2）。

## 0. 前期工作

<img src="https://i.loli.net/2020/11/01/7mrRavDpHeYbWwK.jpg" alt="图1. ROI(Box)" style="zoom: 80%;" />

在ROI(box) Head，我们把1.来自FPN的特征图，3. 提案盒子，4. 标注（ground-truth）盒子作为输入。

1. **来自FPN的特征图**

   正如我们在[第2部分]()所看到的，FPN的输出特征图为：

   ```
   output["p2"].shape -> torch.Size([1, 256, 200, 320]) # stride = 4
   output["p3"].shape -> torch.Size([1, 256, 100, 160]) # stride = 8
   output["p4"].shape -> torch.Size([1, 256, 50, 80]) # stride = 16
   output["p5"].shape -> torch.Size([1, 256, 25, 40]) # stride = 32
   output["p6"].shape -> torch.Size([1, 256, 13, 20]) # stride = 64
   ```

   每个张量尺寸代表（批次、通道、高度、宽度）。我们在整个博客系列中都使用上面的特征尺寸。P2-P5的特征被送入盒头，P6不用。

2. RPN的输出实例中包括**提案框**（见[第4部分]()），其中有1000个\"提案框\"和1000个\"objectness_logits\"。在ROI头中，只有提案框用于裁剪特征图和处理ROI，而objectness\_logits没有被使用。

   ```json
   {
       'proposal_boxes':
   		Boxes(tensor([[675.1985, 469.0636, 936.3209, 695.8753],
   					  [301.7026, 513.4204, 324.4264, 572.4883],
   					  [314.1965, 448.9897, 381.7842, 491.7808],
   					  ...,
   	'objectness_logits':
   		tensor([ 9.1980, 8.0897, 8.0897, ...] 
   }
   ```

3. 已从数据集中加载了**标注框**（见[第三部分]()）

   ```json
   'gt_boxes':
   	Boxes(tensor([[100.55, 180.24, 114.63, 103.01],
   				  [180.58, 162.66, 204.78, 180.95]])),
   'gt_classes': tensor([9, 9]) 
   ```
   
   图3为ROI HEAD的详细示意图。所有的计算都在Detectron2的GPU上进行。

<img src="https://i.loli.net/2020/11/01/A9dWDbcQmypBC5K.jpg" alt="图3.ROI Head 的示意图。蓝色和红色标签分别代表类名称和章节标题。" style="zoom:67%;" />



## 1. 提案框抽样 Proposal Box Sampling

(仅在训练期间)

在RPN中，我们从FPN特征的五个层次（P2到P6）中得到了1000个提案框。

提案框用于从特征图中裁剪出感兴趣的区域（ROI），并将其反馈给框头。为了加快训练速度，[在预测的提案中加入了ground-truth框](https://github.com/facebookresearch/detectron2/blob/1f6ebff69b79f93d69e59eca9c2e84a9f03d850e/detectron2/modeling/proposal_generator/proposal_utils.py#L8-L57)。例如，如果图像有两个ground-truth框，提案总数将为1002个。

在训练过程中，首先对前景和背景提案框进行重新采样，以平衡训练目标。

通过使用[Matcher](https://github.com/facebookresearch/detectron2/blob/bd92fe82be3fab3fb9e3092d6f2ff736e432acb6/detectron2/modeling/matcher.py#L8-L126)（见图4），**将IoUs高于[阈值的](https://github.com/facebookresearch/detectron2/blob/654e2f40b07f40c9cb7be2e0c2266a59a7c9f158/detectron2/config/defaults.py#L251)提案作为前景，其他提案作为背景**。请注意，在ROI Heads中，与RPN不同，没有\"忽略（ignored）"框。添加的ground-truth框与自己完全匹配，因此被算作前景。

<img src="https://i.loli.net/2020/11/01/AlDswJWg9GXFb4j.png" alt="图4.Matcher确定锚点对ground-truth盒的分配。该表显示了IoU矩阵，其形状为（GT盒数，锚点数）。" style="zoom:50%;" />

接下来，我们要平衡前景框（foreground box）和背景框（background truth）的数量。让*N*是（前景+背景）框的目标数量，F是前景框的目标数量。*N*和*F* / *N*由以下[配置参数](https://github.com/facebookresearch/detectron2/blob/bd92fe82be3fab3fb9e3092d6f2ff736e432acb6/detectron2/config/defaults.py#L253-L259)定义。如图5所示，对盒子进行采样，使前景盒子的数量小于*F*。

```python
N：MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE（通常为512）
F/N：MODEL.ROI_HEADS.POSITIVE_FRACTION （通常为0.25
```



## 2. ROI 池化Pooling

ROI池化过程会对提案框指定的特征图的矩形区域进行裁剪（或池化）。



1. **level assignment**

   假设我们有两个提案框（图6中的灰色和蓝色矩形），特征图P2到P5。

   <u>每个方框应该从哪个特征图上裁剪一个ROI</u>？如果你把小灰框分配给P5特征，那么框内只包含一两个特征像素，这并不具有信息量。

   **有一个规则，将提案框分配给相应的feature map。**

   分配的特征级别： floor(4+log2(sqrt(box\_area)/224))

   其中224是**规范的**框体大小。例如，如果提案框的尺寸为224*×*224，则分配到第四级（P4）。

   在图6中，灰色的方框被分配到P2层，蓝色的方框被分配到P5层。级别分配在 [assign_boxes_to_levels 函数中进行](https://github.com/facebookresearch/detectron2/blob/4fa6db0f98268b8d47b5e2746d34b59cf8e033d7/detectron2/modeling/poolers.py#L14-L47)。<img src="https://i.loli.net/2020/11/01/3rIkXWGKDs7R15E.png" alt="图6.投资回报率池的提案框的特征级分配" style="zoom: 60%;" />



2. **ROIAlignV2**

   为了通过具有浮点坐标的提案框准确裁剪ROI，在Mask R-CNN论文中提出了一种名为<u>ROIAlign的方法</u>。在Detectron 2中，默认的池化方法叫做ROIAlignV2，也就是ROIAlign的略微修改版。

   在图7中，描述了ROIAlignV2和ROIAlign。一个大的矩形是ROI中的一个bin（或像素）。为了汇集矩形内的特征值，四个采样点被放置在四个相邻像素值的插值。最终的bin值是通过对四个采样点的值进行平均来计算的。ROIAlignV2和ROIAlign的区别很简单。[从ROI坐标中减去半像素偏移量](https://github.com/facebookresearch/detectron2/blob/806d9ca771a449d5db6265462bda5f36c6752043/detectron2/layers/csrc/ROIAlign/ROIAlign_cpu.cpp#L140-L145)，以更准确地计算相邻像素指数。详情请看图7。

   <img src="https://i.loli.net/2020/11/01/dRHpkNVQCvr2g9M.png" alt="图7. ROIAlignv2.ROIAlignv2。与ROIAlign(v1)相比，从ROI坐标中减去半像素偏移量(0.5)，以更准确地计算相邻像素指数。ROIAlignV2采用像素模型，像素坐标代表像素的中心。" style="zoom: 50%;" />

   得到的张量大小为：

   [B，C，H，W\]=\[N*×*batch size，256，7，7\]

   其中B、C、H、W分别代表整个批次的ROI数量、通道数、高度和宽度。默认情况下，[一个批次N的ROI数量为512](https://github.com/facebookresearch/detectron2/blob/4fa6db0f98268b8d47b5e2746d34b59cf8e033d7/detectron2/config/defaults.py#L253-L257)，[ROI大小为7×7](https://github.com/facebookresearch/detectron2/blob/4fa6db0f98268b8d47b5e2746d34b59cf8e033d7/configs/Base-RCNN-FPN.yaml#L27)。张量是裁剪后的实例特征的集合，其中包括平衡前景和背景ROI。



​		

## 3. Box Head

ROI Pooling后，裁剪后的特征会被送入到头网络中。至于Mask R-CNN，有两种Head：Box Head 和Mask Head。然而Base R-CNN FPN只有[BoxHead，名为FastRCNNConvFCHead](https://github.com/facebookresearch/detectron2/blob/4fa6db0f98268b8d47b5e2746d34b59cf8e033d7/detectron2/modeling/roi_heads/box_head.py#L23-L109)，它对ROI内的对象进行分类，并对盒子的位置和形状进行微调。

默认情况下，盒头的层数如下。

```python
(box_head).FastRCNNConvFCHead((box_head): FastRCNNConvFCHead(
(fc1):Linear(in_features=12544, out_features=1024, bias=True)
(fc2):Linear(in_features=1024, out_features=1024, bias=True))
(box_predictor)。FastRCNNOutputLayers(
(cls_score):Linear(in_features=1024, out_features=81, bias=True)
(bbox_pred):Linear(in_features=1024, out_features=320, bias=True)
```

如你所见，头部没有包含卷积层。

将大小为\[B，256，7，7\]的输入张量扁平化为\[B，256*×*7×7＝12，544通道\]，送入全连接(FC)层1(fc1)。

经过两个FC层后，张量得到最后的box\_predictor层：cls\_score（线性）和bbox\_pred（线性）。
最终层的输出张量是：

```python
cls_score -> scores # shape:(B, 80+1)
bbox_pred -> prediction_deltas # shape:(B, 80×4)
```

接下来我们看看如何计算训练过程中输出的损失。

## 4. 损失计算

(仅在训练期间)

两个损失函数被应用于最终的输出张量。

### 4.1 [本地化损失 ( loss_box_reg)](https://github.com/facebookresearch/detectron2/blob/cc2d218a572c2bfea4fd998082a9e753f25dee15/detectron2/modeling/roi_heads/fast_rcnn.py#L227-L283)

-   l1损失

-   **前景预测**是从*pred\_proposal\_deltas*张量中挑选出来的，其形状为（N个样本*×*批次大小，80×4）。例如，如果第15个样本是前台，类指数=17，则选取\[14(=15-1)，\[68(=17×4)，69，70，71\]\]的指数。

-   **前景ground truth目标**是从*gt\_proposal\_deltas*中挑选出来的，其形状为（B，4）。张量值是地真盒与提案盒相比的相对大小，由[Box2BoxTransform.get_deltas](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/box_regression.py#L38-L71)函数计算（见[Part4]() 的3-3节）。带有前景指数的张量是从gt\_proposal\_deltas中采样得到的。

### 4.2 [分类损失(loss\_cls)](https://github.com/facebookresearch/detectron2/blob/cc2d218a572c2bfea4fd998082a9e753f25dee15/detectron2/modeling/roi_heads/fast_rcnn.py#L214-L225)

-   软最大交叉熵损失

-   计算所有前景和背景预测得分\[B，K类\]与地面真相类指数\[B\]的关系。

-   分类目标**包括前景类和背景类**，所以K=类数+1（COCO数据集的背景类数为\'80\'）。

下面的损失结果加上RPN中计算的损失——"loss\_rpn\_cls\"和\"loss\_rpn\_cls\"，加起来就是模型的总损失。

```json
{
	'loss_cls': tensor(4.3722, device='cuda:0', grad_fn=< NllLossBackward>),
	'loss_box_reg': tensor(0.0533, device='cuda:0', grad_fn=<DivBackward0>)
}
```



## 5. 推论预测

(只在测试期间)

正如我们在第3节中所看到的，我们有形状为（B，80+1）的*分数*和形状为（B，80×4）的*prediction\_deltas*作为Box Head的输出。

(1) **将预测三角区应用于提案框**

为了从预测的deltas⁶ : Δx, Δy, Δw, 和Δh计算最终的盒子坐标，使用[Box2BoxTransform.apply_deltas](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/box_regression.py#L38-L71)函数（图8）。这[与第4部分第5节中的步骤1]()相同。

![图8.将预测三角应用于提案框，计算出最终预测框的坐标。](https://i.loli.net/2020/11/01/p7NyoPQqLlzbJS9.png)

(2) **按分数筛选框**

我们首先[过滤掉低分的边界框](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/fast_rcnn.py#L107-L115)如图9所示（从左到中）。每个框都有相应的分数，所以很容易做到这一点。

![图9.推理阶段的后处理。在推理阶段的后处理.左：后处理前所有ROI的可视化.中间：分数阈值化后.右：非最大抑制后。](https://i.loli.net/2020/11/01/YPVZnla5xIegFWs.jpg)



(3) **非最大压制**

为了去除重叠的盒子，应用非最大抑制（NMS）（图9，从中间到右边）。[这里](https://github.com/facebookresearch/detectron2/blob/e1356b1ee79ad2e7f9739ad533250e24d4278c30/detectron2/config/defaults.py#L261-L271)定义了NMS的参数。

(4) **择优录取**

最后，当剩余框数超过[预设数时](https://github.com/facebookresearch/detectron2/blob/e1356b1ee79ad2e7f9739ad533250e24d4278c30/detectron2/config/defaults.py#L558-L560)，我们选择top-k的结果。