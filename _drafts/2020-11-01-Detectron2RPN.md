---
title: 【深度学习】Faster-RCNN ：3.核心Region Proposal Network

tags:
  - Detectron2
categories:
  - - 机器学习
    - 原理
---

【深度研究Detectron2】，基于文档Digging into Detectron2。本篇主要对于Region Proposal Network 进行详细解释，主要从其结构、Anchor的生成、表述数据准备、损失计算、提案选择等方面来讲解。

<!-- more -->

本篇将深入到最复杂但最重要的部分——Region Proposal Network（见图1）。

<img src="https://i.loli.net/2020/11/01/V674WylUXmpF2bc.png" alt="图1" style="zoom: 50%;" />

# 0. 主要术语

* **Anchor**：锚点（用于定位object的框框的点）

* **Grid**：网格（图像中各个像素构成的网络）

* **loU**：物体检测需要定位出物体的bounding box。在目标检测时，我们不仅仅需要定位出待检测物体的bounding box，我们还要识别出box内是什么。由于我们算法不可能和人工标注的数据完全的匹配，其中的偏差（或者说是精度），则通过IoU表示（定位精度评价）IoU定义为两个bounding box 的重叠度：

  <img src="https://raw.githubusercontent.com/Dinghye/Figurebed/master/image-20201031145155437.png" alt="image-20201031145155437" style="zoom: 67%;" />

  矩形框A、B的一个重合度IoU计算公式则为：IoU=(A∩B)/(A∪B)即具体表示为：IoU=S_I(S_A+S_B-S_I)Region Proposal方法，比传统华东窗口方法获取方法的质量要更高。比较常用的有：SelectiveSearch（ss，选择性搜索）、Edge Boxes（EB）基于RP目标检测算法的步骤如下：

  ![img](https://img-blog.csdn.net/20161020124321529?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

  边框回归（Bouding Box Regression）：是对RegionProposal进行纠正的线性回归算法，目的是为了让Region Proposal提取到的窗口与目标窗口（Ground Truth）更加吻合

* **foreground**：前景（相对于背景，即所需的目标）

* **box**：箱、盒

* **ground-truth** ：标注数据（确定为真实的标注数据）

* **proposal** ：提案（即可能的结果）

# 1. RPN结构

正如我们在[第2部分](https://dinghye.gitee.io/2020/10/31/Detectron2FPN/)看到的，特征金字塔网络（FPN）的输出特征图为：

```python
output["p2"].shape -> torch.Size([1, 256, 200, 320]) # stride = 4
output["p3"].shape -> torch.Size([1, 256, 100, 160]) # stride = 8
output["p4"].shape -> torch.Size([1, 256, 50, 80]) # stride = 16
output["p5"].shape -> torch.Size([1, 256, 25, 40]) # stride = 32
output["p6"].shape -> torch.Size([1, 256, 13, 20]) # stride = 64
```

同样的，这也是**RPN的输入**。每个张量尺寸代表（batch、通道、高度、宽度）。我们在整个博客部分都使用上面的特征尺寸。

我们也有从数据集加载的标注框（见[第3部分](https://dinghye.gitee.io/2020/11/01/Detectron2DataLoader/)）：

```json
'gt_boxes':Boxes(tensor([
[100.58, 180.66, 214.78, 283.95],
[180.58, 162.66, 204.78, 180.95]
])),
'gt_classes': tensor([9, 9]) # 在RPN中没有使用!
```

检测器如何连接特征图(feature map)和标注框(ground truth)的位置和大小？让我们来看看RPN\--RCNN检测器的核心部件——是如何工作的！图2为RPN的详细示意图。RPN由神经网络（RPN Head）和非神经网络功能组成。RPN³中的所有计算都在Detectron 2的GPU上进行。

<img src="https://i.loli.net/2020/11/01/G6dEMKcH2gqNShw.png" alt="图2.RPN示意图。蓝色和红色标签分别代表类名称和下面的章节名称。" style="zoom:50%;" />

首先，我们来看看处理FPN送来的特征图的RPN头。

# 2. RPN Head

RPN的神经网络部分很简单。它被称为RPN Head，由[标准RPNHead类中](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/proposal_generator/rpn.py#L34-L85)定义的三个卷积层组成。

1. *cov* (3×3，256-\>256通道)
2. *objectness logits conv*（1×1，256 -\> 3 ch）
3. *anchor deltas conv* (1×1，256->3×4 ch）

五级（P2～P6）的特征图逐一[送入网络](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/proposal_generator/rpn.py#L74-L85)。

一级的输出特征图为：

1. *pred_objectness_logits*（B，3 ch，Hi，Wi）：对象存在的概率图。

2. *pred_anchor_deltas*(B，3×4 ch，Hi，Wi)：与锚的相对箱形。

   其中，B代表批量大小，Hi和Wi对应P2到P6的特征图大小。

它们实际上是什么样子的呢？在图4中，将每一级的objectness logits map叠加在输入图像上。你可以发现，**小的物体在P2和P3被检测到，大的物体在P4到P6**。这正是特征金字塔网络的目标。<u>多尺度网络可以检测到单尺度检测器无法发现的微小物体。</u>

接下来，我们继续进行锚点生成，这对于将标注框与上面两个输出特征图关联起来是至关重要的。

<img src="https://i.loli.net/2020/11/02/f7BlL5viXhPO4Mj.png" alt="图3.objectness maps 的可视化。Sigmoid函数已被应用到objectness_logits map上。1:1 anchor的objectness maps 被调整为P2特征图大小，并覆盖在原始图像上。" style="zoom: 80%;" />

# 3. 锚点生成 Anchor Generation

为了将objectness map和 anchor deltas map与标注框连接起来，需要称为\"锚（anchor）"的参考框。

### 3.1 生成cell anchor

在[Detectron 2的Base-FPN-RCNN](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/configs/Base-RCNN-FPN.yaml#L9-L11)中，Anchor的定义如下：

```python
MODEL.ANCHOR_GENERATOR.SIZEES = [[32], [64], [128], [256], [512]]
MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]] 。
```

>  **它的含义是什么？**

ANCHOR\_GENERATOR.SIZEES列表中的五个元素对应五个级别的特征图（P2到P6）。例如P2（stride=4）有一个anchor，其大小为32

长宽比定义了锚的形状。在上面的例子中，有三种形状：0.5、1.0和2.0。让我们看看实际的锚点（图5）。P2特征图处的三个锚点的长宽比分别为1:2、1:1和2:1，面积与32×32相同。在P3级别的锚是P2锚的两倍。

![图4. P2和P3特征图的单元锚。P2和P3特征图的单元锚。(左起：1:2、1:1和2:1纵横比)](https://i.loli.net/2020/11/01/ZFsKWlCAQNrHqIa.png)



这些锚在Detectron 2中称为\"单元锚(cell anchor)"。(锚生成的代码在[这里](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/anchor_generator.py#L140-L177)) 结果我们得到了5个特征图级别的3×5=15个cell锚。

P2、P3、P4、P5和P6的cell anchor。(x1, y1, x2, y2)

```python
tensor([[-22.6274，-11.3137，22.6274，11.3137],
[-16.0000, -16.0000, 16.0000, 16.0000],
[-11.3137, -22.6274, 11.3137, 22.6274]])

tensor([[-45.2548，-22.6274，45.2548，22.6274],
[-32.0000, -32.0000, 32.0000, 32.0000],
[-22.6274, -45.2548, 22.6274, 45.2548]])

tensor([[-90.5097，-45.2548，90.5097，45.2548],
[-64.0000, -64.0000, 64.0000, 64.0000],
[-45.2548, -90.5097, 45.2548, 90.5097]])

tensor([[-181.0193，-90.5097，181.0193，90.5097]。
[-128.0000, -128.0000, 128.0000, 128.0000],
[ -90.5097, -181.0193, 90.5097, 181.0193]])

tensor([[-362.0387，-181.0193，362.0387，181.0193]。
[-256.0000, -256.0000, 256.0000, 256.0000],
[-181.0193, -362.0387, 181.0193, 362.0387]])
```

### 3.2 在网格上放置锚点

接下来我们将单元格锚放在网格上，其大小与预测的特征图相同。

例如我们预测的特征图\'P6\'的大小为（13，20），步幅为64。在图6中，P6网格的三个锚点放置在（5，5）。在输入图像分辨率中，(5，5)的坐标对应于(320，320)，方块锚的大小为(512，512)。锚被放置在每个网格点上，所以13×20×3=780个锚被生成在P6。
对其他网格进行同样的处理（见图6中P5网格的例子），总共生成255,780个锚。

<img src="https://i.loli.net/2020/11/01/MlLu8oSAdaVeXTJ.png" alt="图6.在网格上放置锚。在网格上放置锚点。每个网格的左上角对应于（0，0)" style="zoom:45%;" />

# 4. 标注数据准备

在本章中，我们将标注框与生成的锚联系起来。

### 4.1 计算交叉点-单位（IoU）矩阵

假设我们从一个数据集中加载了两个ground-truth（GT）box（已经标注出的数据框）。

```json
'gt_boxes'：
	Boxes(tensor([
		[100.58, 180.66, 214.78, 283.95],
		[180.58, 162.66, 204.78, 180.95]
	])),
```

我们现在试着从255780个anchor中找出与这两个标注盒子相似的盒子。如何判断一个盒子与另一个盒子是否相似呢？答案是**进行IoU计算**。（什么是IoU？见0.1）在Detectron2中，[pairwise_iou](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/structures/boxes.py#L299-L331)函数可以从**两个盒子列表中计算出每一对盒子的IoU**。在我们的例子中，pairwise\_iou的结果是一个**大小为(2(GT), 255780(anchors))的矩阵**。

```python
# IoU矩阵例子，pairwise_iou的结果。
tensor([[0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000], #GT 1
[0.0000, 0.0000, 0.0000, ..., 0.0087, 0.0213, 0.0081], #GT 2
```



### 4.2 用Matcher检查IoU矩阵

Matcher对IoU矩阵进行检查，所有的锚都被标记为前景、背景或忽略。如图7所示，如果IoU大于[预先定义的阈值](https://github.com/facebookresearch/detectron2/blob/989f52d67d05445ccd030d8f13d6cc53e297fb91/detectron2/config/defaults.py#L199)（通常为0.7），则锚被分配到GT框之一，并标记为前景（\'1\'）。如果IoU小于另一个阈值(通常为0.3)，则将锚标记为背景(\'0\')，否则忽略(\'-1\')。

<img src="https://i.loli.net/2020/11/01/ubewzdoOkPUsDV2.png" alt="图7.Matcher确定锚点对地真盒的分配。该表显示了IoU矩阵，其形状为（GT箱数，锚点数）。" style="zoom: 50%;" />



在图8中，我们展示了叠加在输入图像上的匹配结果。可以看到，大部分网格点被标注为背景（0），少数网格点被标注为前景（1）和忽略（-1）。

<img src="https://i.loli.net/2020/11/01/MGSLfFRtPTijWwx.png" alt="图8.在输入图像上叠加匹配结果。匹配结果叠加在输入图像上。" style="zoom: 67%;" />



### 4.3 计算锚点三角

被确定为前景的锚箱与GT箱的形状相似。然而，网络应该学习提出GT盒的确切位置和形状。为了实现这一目标，应该学习四个回归参数：Δx、Δy、Δw和Δh。这些\"deltas\"的计算方法如图9所示，使用[Box2BoxTransform.get_deltas](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/box_regression.py#L38-L71)函数。该公式写在Faster-RNN论文中。

![图9. 锚点三角计算](https://i.loli.net/2020/11/01/28QfrJEYThX97zt.png)

结果，我们得到一个名为*gt\_anchor\_deltas*的张量，在我们的情况下，它的形状是（255,780，4）。

```python
# 计算出的三角(dx, dy, dw, dh)
tensor([[[9.9280, 24.6847, 0.8399, 2.8774],
[14.0403, 17.4548, 1.1865, 2.5308],
[19.8559, 12.3424, 1.5330, 2.1842],
...,
```



### 4.4 重新取样计算损失的boxes

现在我们在特征图的每个网格点上都有objectness\_logits和anchor\_deltas，我们可以将预测的特征图与之进行比较。

* 问题：

  图10(左)是每张图片的锚点数量和例子的分类。如你所见，大部分的锚都是背景锚。例如，通常在255,780个锚中，foreground 锚不到100个，被忽略的锚不到1000个，其余的都是背景锚。如果我们继续训练，由于标签不平衡，将会很难学会foreground（这里forground指的目标）。

* 解决：

  通过使用[subsample_labels函数](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/sampling.py#L7-L50)对标签进行重新采样，解决不平衡问题。

设*N*为前景+背景框的目标数量，*F*为前景框的目标数量。*N*和*F* / *N*由以下配置参数定义。

```python
N：MODEL.RPN.BATCH_SIZE_PER_IMAGE（通常为256）。
F/N：MODEL.RPN.POSITIVE_FRACTION(通常为0.5)
```

图10（中心）显示了重新采样框的分解。背景箱和前景箱是**随机选择**的，因此*N*和*F/N*成为上述参数定义的值。在前景数小于*F的*情况下，如图10(右)所示，背景框被采样来填充*N个*样本。

<img src="https://i.loli.net/2020/11/01/xb1UKlZci3yPkTg.png" alt="图10.对前景和背景框进行重新采样。重新对前景和背景框进行采样" style="zoom:50%;" />

# 5. 损失计算

在[rpn_losses函数](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/detectron2/modeling/proposal_generator/rpn_outputs.py#L159-L195)处对预测图和ground-truth map应用了两个损失函数。

### 5.1 本地化损失 (loss\_rpn\_loc)

-   l1损失

-   **只在ground-truth objectness=1（前景）的网格点上**计算，也就是忽略所有背景网格来计算损失。

### 5.2 客观性损失(损失\_rpn\_cls)

-   二元交叉熵损失。

-   **只在ground-truth objectness=1（前景）或0（背景）的网格点上**计算。

实际损失结果如下：

```python
{
	'loss_rpn_cls': tensor(0.6913, device='cuda:0', grad_fn=<MulBackward0>)
	'loss_rpn_loc': tensor(0.1644, device='cuda:0', grad_fn=<MulBackward0>)
}
```

# 6. 提案选择

最后，我们按照下面的四个步骤从预测框中选择1000个\"区域提案\"框。

1.  [将预测的anchor_deltas应用](https://github.com/facebookresearch/detectron2/blob/e1356b1ee79ad2e7f9739ad533250e24d4278c30/detectron2/modeling/box_regression.py#L73-L110)到相应的锚中，这是3-3的反向过程。

2.  预测框按各特征层次的预测对象性得分排序。

3.  如图11所示，从图像的每个特征级别⁶中选择前K个得分的方框（由[配置参数](https://github.com/facebookresearch/detectron2/blob/5e2a1ecccd228227c5a605c0a98d58e1b2db3640/configs/Base-RCNN-FPN.yaml#L14-L20)定义）。例如，从P2的192,000个盒子中选择2000个盒子。对于P6存在少于2000个盒子的地方，则选择所有的盒子。

4.  非最大抑制([batched_nms](https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/nms.py#L9-L26))在每个级别独立应用。结果有1,000个得分最高的盒子保留下来。

<img src="https://i.loli.net/2020/11/01/KdY548NfrzIPmkp.png" alt="图11.从每个特征层中选择顶K提案箱。从每个特征级别中选择top-K提案框。框的数量是输入图像大小为（H=800，W=1280）时的例子。" style="zoom: 67%;" />



最后，我们以\'实例\'的形式获得提案盒子：

```json
'proposal_boxes': 1,000个盒子
'objectness_logits': 1,000个分数
```

这些分数将在下一阶段使用。

