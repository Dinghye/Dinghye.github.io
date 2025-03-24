---
title: 【深度学习】Detectron2& Faster-RCNN指北

tags:
  - Detectron2
categories:
  - - 机器学习
    - 原理
---

Detectron2 系列 Digging into Detectron2 笔记索引。

<!-- more-->

# 1. 简介

本系列主要讲解Detectron2 中Faster RCNN的原理，简单介绍了Detectron2的结构和工作机制。共有五个部分：

* 第一部分-[简介：基本网络架构和Repo结构](https://dinghye.gitee.io/2020/10/31/Detectron2Total/)：主要介绍了什么是Dectron2，以及它的基本结构
* 第二部分-[特征金字塔网络 Feature Pyramid Network](https://dinghye.gitee.io/2020/10/31/Detectron2FPN/)：主要介绍了在Dectectron2 中使用的Backbone network——Feature Pyramid Network（FPN）的原理、作用和结构进行了简单介绍
* 第三部分-[数据加载注册原理](https://dinghye.gitee.io/2020/11/01/Detectron2DataLoader/)：主要介绍了Ground-truth Data以及在Detectron2 中的作用，同时介绍了在Detetron2 中如何对数据进行注册以及使用。
* 第四部分-[核心Region Proposal Network](https://dinghye.gitee.io/2020/11/01/Detectron2RPN/)：主要介绍了Region Proposal Network的作用，Anchor的生成
* 第五部分-[ROI(Box) Head](https://dinghye.gitee.io/2020/11/01/Detectron2ROI/)：主要介绍了ROI(Box) Head

文章都给出 了对应术语和含义



# Faster-RCNN

Faster R-CNN可以简单地看做“区域生成网络RPNs + Fast R-CNN”的系统，用区域生成网络代替FastR-CNN中的Selective Search方法。Faster R-CNN着重解决了这个系统中的三个问题：

1. 如何**设计**区域生成网络；
2. 如何**训练**区域生成网络；
3. 如何让区域生成网络和Fast RCNN网络**共享特征提取网络** 

其主要特点为：

| 类型                                                         | **使用方法**                                                 | **缺点**                                                     | **改进**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| R-CNN (Region-based Convolutional Neural Networks)           | 1、SS提取RP； 2、CNN提取特征； 3、SVM分类； 4、BB盒回归。    | 1、 训练步骤繁琐（微调网络+训练SVM+训练bbox）； 2、 训练、测试均速度慢 ； 3、 训练占空间 | 1、 从DPM HSC的34.3%直接提升到了66%（mAP）； 2、 引入RP+CNN  |
| Fast R-CNN (Fast Region-based Convolutional Neural Networks) | 1、SS提取RP； 2、CNN提取特征； 3、softmax分类； 4、多任务损失函数边框回归。 | 1、 依旧用SS提取RP(耗时2-3s，特征提取耗时0.32s)； 2、 无法满足实时应用，没有真正实现端到端训练测试； 3、 利用了GPU，但是区域建议方法是在CPU上实现的。 | 1、 由66.9%提升到70%； 2、 每张图像耗时约为3s。              |
| Faster R-CNN (Fast Region-based Convolutional Neural Networks) | **1、RPN提取RP； 2、CNN提取特征； 3、softmax分类； 4、多任务损失函数边框回归。** | 1、 还是无法达到实时检测目标； 2、 获取region proposal，再对每个proposal分类计算量还是比较大。 | 1、 提高了检测精度和速度； 2、 真正实现端到端的目标检测框架； 3、 生成建议框仅需约10ms。 |