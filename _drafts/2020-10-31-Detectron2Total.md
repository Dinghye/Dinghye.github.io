---
title: 【Detectron2】简介：基本网络架构和Repo结构

tags:
  - Detectron2
categories:
  - - 机器学习
    - 原理
---

【深度研究Detectron2】，基于文档Digging into Detectron2。本篇主要对Detectron2 进行总体结构的概述。

<!--more -->

# 1. Detectron 2是什么？

Detectron 2 ²是Facebook AI Research的下一代开源对象检测系统。通过该repo，您可以使用和训练各种最先进的模型，用于检测任务，如边界框检测、实例和语义分割以及人的关键点检测。

你可以按照版本库的说明——[安装](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)和[入门](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md)——来运行演示，但如果你想走得更远，而不仅仅是运行示例命令，那就需要了解版本库的工作原理。

# 2. Faster R-CNN FPN架构

作为一个例子，我选择了以特征金字塔网络（Feature Pyramid Network³）（[BASE-RCNN-FPN](https://github.com/facebookresearch/detectron2/blob/master/configs/Base-RCNN-FPN.yaml)）为基础的 Faster R-CNN，它是基本的边界盒探测器，可扩展到Mask R-CNN。以[FPN](https://arxiv.org/abs/1612.03144)为骨架的Faster R-CNN探测器是一种多尺度的探测器，实现了从微小物体到大物体的高精度检测，使自己成为事实上的标准探测器（见图1）。

BASE R-CNN FPN的结构如下：

![图1.R-CNN与特征金字塔网络的推断结果Faster（Base）R-CNN与特征金字塔网络的推断结果。](https://i.loli.net/2020/10/31/DpLKSirwU4JPCZ3.png)



![图2.Base RCNN FPN的元架构。基础RCNN FPN的元架构。](https://i.loli.net/2020/10/31/PIrno7CqeUHWTZk.png)



上面的示意图显示了网络的元架构。你可以看到里面有*三块*，分别是。

1.  [**Backbone Network**](https://dinghye.gitee.io/2020/10/31/Detectron2FPN/)：<u>从输入图像中提取不同比例的特征图</u>。Base-RCNN-FPN的输出特征称为P2（1/4比例）、P3（1/8）、P4（1/16）、P5（1/32）和P6（1/64）。请注意，非FPN(\'C4\')架构的输出特征只是从1/16比例。
2.  [**Region Proposal Network**](https://dinghye.gitee.io/2020/11/01/Detectron2RPN/)：<u>从多尺度特征中检测对象区域</u>。可获得1000个带置信度分数的Proposal Box(提案框)（默认情况下）。
3.  [**Box head**](https://dinghye.gitee.io/2020/11/01/Detectron2ROI/)：<u>将使用提案框的特征图裁剪和扭曲成多个**固定大小**的特征，并通过全连接层获得微调的箱体位置和分类结果。</u>最后利用非最大抑制（NMS）过滤掉最大的100个盒子（默认）。框头是**ROI Heads**的子类之一。例如Mask R-CNN有更多的ROI头，如Mask Head。

每个块里面都有什么？图3显示了详细的架构。

<img src="https://i.loli.net/2020/10/31/GdW6vBLaNIF4Zft.png" alt="图3.BASE-RCNN-FPN的详细架构。Base-RCNN-FPN的详细架构。蓝色标签代表类名。" style="zoom: 67%;" />

后面的文章将详细对其中的每一个部分进行讲解。

# 3. Detectron2 repo的结构

以下是 detectron 2 的目录树（在detectron2目录下⁶）。请看‘modeling’目录即可。Base-RCNN-FPN架构是由该目录下的几个类构建的。

<img src="https://i.loli.net/2020/10/31/wAbTZoyaOkFUzBL.png" alt="目录" style="zoom: 67%;" />

**Meta Architecture 元架构**
[GeneralizedRCNN](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/rcnn.py) (meta\_arch/rcnn.py)，它有：

1.  **Backbone Network**
    [FPN](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/backbone/fpn.py) (backbone/fpn.py)
    └ [ResNet](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/backbone/resnet.py) (backbone/resnet.py)

2.  **Region Proposal Network**。
    [RPN](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/proposal_generator/rpn.py)(proposal\_generator/rpn.py)
    ├[标准RPNHead](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/proposal_generator/rpn.py)(proposal\_generator/rpn.py)
    └[RPNOutput](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/proposal_generator/rpn_outputs.py)(proposal\_generator/rpn\_outputs.py)

3.  **ROI Heads(Box Head)**。
    [标准ROIHeads](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/roi_heads.py) (roi\_heads/roi\_heads.py)
    ├[ROIPooler](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/poolers.py) (poolers.py)
    ├[FastRCNNConvFCHead](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/box_head.py) (roi\_heads/box\_heads.py)
    ├[FastRCNNNOutputLayers (](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/fast_rcnn.py)roi\_heads/fast\_rcnn.py)
    └[FastRCNNNOutputs](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/fast_rcnn.py) (roi\_heads/fast\_rcnn.py)

每个区块都有一个主类和子类。

现在请看图3上的*蓝色标签*。你可以看到哪个类对应于pipeline的哪个部分。在这里，我添加了没有类名的架构图。

<img src="https://i.loli.net/2020/10/31/MvypIr2OYudwnKx.png" alt="图4.BASE-RCNN-FPN的详细架构（无类名） Base-RCNN-FPN的详细架构（无类名）。" style="zoom: 50%;" />

