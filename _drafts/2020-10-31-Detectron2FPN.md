---
title: 【深度学习】Faster-RCNN：1. 特征金字塔网络Feature Pyramid Network

tags:
  - Detectron2
categories:
  - - 机器学习
    - 原理
mathjax: true
---

【深度研究Detectron2】，基于文档Digging into Detectron2。本篇主要对Detectron2中骨干网络（Backbone network）——特征金字塔网络Feature Pyramid Network的架构以及原理简单介绍。你将了解到：**Backbone network、 FPN、ResNet**的相关知识

<!--more-->

* 先修知识：

  BP反向神经网络：https://blog.csdn.net/sunny_yeah_/article/details/88560830

  Detectron2系列part1：[简介：基本网络架构和Repo结构](https://dinghye.gitee.io/2020/10/31/Detectron2Total/)

* 参考链接：

  [FPN（feature pyramid networks）算法讲解](https://blog.csdn.net/u014380165/article/details/72890275)

  [ResNet详解——通俗易懂版](https://blog.csdn.net/sunny_yeah_/article/details/89430124)

  [backbone、head、neck等深度学习中的术语解释]([https://blog.csdn.net/t20134297/article/details/105745566)

  [什么是张量（tensor）](https://www.zhihu.com/question/20695804/answer/43265860)



这一部分，我们将深入了解骨干网络--特征金字塔网络³（FPN）。

# 0. FPN基础知识

这一部分主要对下面讲到的专业术语进行了详细的解释，只看FPN的可以跳过这一部分

## 0.1 Backbone network、FPN、ResNet

1. **Backbone** 翻译为骨干网络。是网络的一部分，这一部分通常来说用于**提取特征**，比如提取图片中的信息（feature map）给后面的网络使用。

   > Backbone network骨干网络的作用是**从输入图像中提取特征图**。

   在深度学习中担任这类工作的都可以叫做Backbone network，常见的比如resnet VGG等

2. **FPN（特征金字塔网络）**：这是具体的一种Backbone网络，其特点是能够对**多尺度**的特征信息进行处理。原来多数的object detection算法都是只采用<u>顶层特征做预测</u>，但我们知道低层的特征语义信息比较少，但是目标位置准确；高层的特征语义信息比较丰富，但是目标位置比较粗略。另外虽然也有些算法采用多尺度特征融合的方式，但是一般是采用融合后的特征做预测，而FPN不一样的地方在于预测是在不同特征层独立进行的。（原文地址：https://arxiv.org/abs/1612.03144）特别的，在原文当中FPN指的是包含FPN的检测网络，而这里只用到了它的FPN。（概念不混淆）

3. **ResNet**：ResNet是一种残差网络，咱们可以把它理解为一个子网络，这个子网络经过堆叠可以构成一个很深的网络。其结构如下：

   <img src="https://i.loli.net/2020/11/02/8FyU1GnuJg6H7aT.png" alt="图1. ResNet块" style="zoom:50%;" />

   * 为什么要引入ResNet？

     我们知道，网络越深，咱们能获取的信息越多，而且特征也越丰富。但是根据实验表明，随着网络的加深，优化效果反而越差，测试数据和训练数据的准确率反而降低了。**这是由于网络的加深会造成梯度爆炸和梯度消失的问题。**目前针对这种现象已经有了解决的方法：对输入数据和中间层的数据进行归一化操作，这种方法可以保证网络在反向传播中采用随机梯度下降（SGD），从而让网络达到收敛。但是，这个方法仅对几十层的网络有用，当网络再往深处走的时候，这种方法就无用武之地了。ResNet！登场

   * ResNet如何解决问题？

     下图为ResNet的两种结构

     <img src="https://i.loli.net/2020/11/02/csmEkxfAr4hY9wd.png" alt="图2. 两种ResNet block" style="zoom: 67%;" />

     我们先想一下，为什么越深会导致训练结果越差呢？我们假设我们需要求解的映射是$H(x)$ ，H(x)的计算结果为观测值，$x$是估计值。这个时候，我们的优化方向，应该就是想办法让$H(x)=x$(观测和估计值一致)。但是这个情况下，我们是不是很难找到多少层比较好，保证它不会下降？

     那么这个时候，我们就引入**残差**帮助解决这个问题。根据前面的假设，残差应该计算为$F(x)=H(x)-x$。我们最后的结果求解的问题就变成了$H(x)=F(x)+x$。我们求解的东西就变成了F(x)。

     欸这个有什么用呢？假如当前我们需要的已经到了最优的情况，我们再往后走，是不是只需要让F(x)靠近0就好了？这样就保证了，在下一层中，我们仍然能保证它的最优。

     > 这里我们称x为identity Function，它是一个条约链接，F(x)为ResNet Function

     其公式为：(两层)
     $$
     a^{[l+2]}=Relu(W^{[l+2]}(Relu(W^{[l+1]}a^{[l]}+b^{[l+1]})+b^{[l+2]}+a^{[l]})
     $$
     

## 0.2 其他基础术语

在讲解网络结构中还出现了其他术语如下有：

1. **head：**head是获取网络输出内容的网络，利用之前提取的特征，head利用这些特征，做出预测

2. **neck：**是放在backbone和head之间的，是为了更好的利用backbone提取的特征

3. **bottleneck：**瓶颈的意思，通常指的是<u>网络输入的数据维度和输出的维度不同，输出的维度比输入的小了许多</u>，就像脖子一样，变细了。经常设置的参数 bottle_num=256，指的是网络输出的数据的维度是256 ，可是输入进来的可能是1024维度的。

4. **tensor** ：张量

   * 什么是张量？

     > A tensor is something that transforms like a tensor！一个在不同的参考系下按照某种特定的法则进行变换，就是张量

   * 为什么要用张量

     物理的定律是不会随着参考系变化而发生变化的。考虑下面一个物理过程: 两个粒子1和2经过散射变成了3和4. 在 Andrew 看来, 能动量守恒是!$ (E_{a,1},p_{a,1})+(E_{a,2},p_{a,2})=(E_{a,3},p_{a,3})+(E_{a,4},p_{a,4}) $但这样写, 并不能直接看出 Bob 也看到能动量守恒. 但如果用张量的语言直接写成: $ T_1+T_2=T_3+T_4 $, 我们立刻就知道它在 Andrew 看来是$(E_{a,1},p_{a,1})+(E_{a,2},p_{a,2})=(E_{a,3},p_{a,3})+(E_{a,4},p_{a,4}) $, 在 Bob 看来是$ (E_{b,1},p_{b,1})+(E_{b,2},p_{b,2})=(E_{b,3},p_{b,3})+(E_{b,4},p_{b,4}) $。 **用张量语言描述的物理定律自动保证了不随参考系变化的这一性质. 而且从记号的角度看, 用张量也更加简洁.**

   * 什么是张量？Deeper！

     在数学线性代数中，线性变换的这个概念的精髓在于，它**不依赖于线性空间的基的选取**。在某一组基下，它的矩阵表示A是一个木有，在另外一组基下，它的矩阵表示它的矩阵表示$ A'=TAT^(-1)$是另一个模样, 其中！$ T $是基变换矩阵. 有一种常见的说法: **矩阵的意义是线性变换, 相似矩阵是同一个线性变换在不同的基下的表示。**借用这个概念，慢着! "同一个线性变换在不同的基下的表示", 这难道不就是和之前说的张量是一回事嘛! Lorentz 变换就是 Minkowski 空间中的基变换, 能动量张量实质上就是一个线性变换. Andrew 和 Bob 看到的能动量张量, 不就是这个线性变换在不同的基下的表示吗？

   * 深度学习与张量

     在深度学习里，**Tensor实际上就是一个多维数组（multidimensional array）**。而Tensor的目的是**能够创造更高维度的矩阵、向量**。

     <img src="https://pic2.zhimg.com/v2-91fefb9a6227e8c11f8df316bc30cbb5_r.jpg" alt="图1. 张量解释（知乎@恒仔）" style="zoom: 33%;" />

# 1. FPN的输入和输出

首先我们要明确FPN的输入和输出。图1是FPN的详细原理图。

<img src="https://i.loli.net/2020/10/31/2QKEGo4MUw8RyHL.jpg" alt="图2.BASE-RCNN-FPN与ResNet50主干网的详细架构。采用ResNet50的Base-RCNN-FPN骨干网的详细架构。蓝色标签代表类名。块内的(a)、(b)和(c)代表瓶颈类型，详见图5。" style="zoom:80%;" />



1. **输入(torch.Tensor) (B, 3, H, W)图像**

   B、H、W分别代表批次大小、图像高度和宽度。<u>注意输入颜色通道的顺序是蓝、绿、红（BGR）。如果将RGB图像作为输入，检测精度可能会下降。</u>（对应在实验的时候也要注意！见4.2节踩坑：[用自己的数据训练Faster-RCNN模型](https://dinghye.gitee.io/2020/10/30/detectron2guidance/)）

2. **输出（dict of torch.Tensor）(B,C,H/S,W/S) feature map 特征图**

   C和S代表通道大小和步长。默认情况下，C=256代表所有刻度，S=4、8、16、32和64分别代表P2、P3、P4、P5和P6输出。

   例如，如果我们将一张尺寸为（H=800，W=1280）的单幅图像放入骨干中，输入的张量尺寸为torch.Size([1，3，800，1280\])，输出的dict应该是：

   ```python
   output["p2"].shape -> torch.Size([1, 256, 200, 320]) # stride = 4
   output["p3"].shape -> torch.Size([1, 256, 100, 160]) # stride = 8
   output["p4"].shape -> torch.Size([1, 256, 50, 80]) # stride = 16
   output["p5"].shape -> torch.Size([1, 256, 25, 40]) # stride = 32
   output["p6"].shape -> torch.Size([1, 256, 13, 20]) # stride = 64
   ```

   图3显示了实际输出特征图的样子。“P6\"特征的一个像素对应着比\"P2\"更广的输入图像区域——换句话说，\"P6\"比\"P2\"有更大的**接受场**。（其实从卷积的角度上很好理解）FPN可以提取具有不同接受场的<u>多尺度特征图</u>。

<img src="https://miro.medium.com/max/1203/1*7z4DmOf-F4KIlJqbqaKxgw.png" alt="图3：FPN的输入和输出示例。从每个输出端可以看到第0通道处的特征" style="zoom: 80%;" />

​			

# 2. FPN的结构

FPN包含ResNet、横向和输出卷积层、上采样器和最后一层maxpool层。[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/fpn.py#L16-L152)

1. **ResNet：**（什么是ResNet？见0.1）

   ResNet由stem block和包含多个bottleneck块的"阶段"组成。在ResNet50中，块状结构为：

   ```python
   BasicStem
   (res2 stage, 1/4 scale)
   BottleneckBlock (b)(stride=1, with shortcut conv)
   BottleneckBlock (a)(stride=1, w/0  shortcut conv) × 2
   
   (res3 stage, 1/8 scale)
   BottleneckBlock (c)(stride=2, with shortcut conv)
   BottleneckBlock (a)(stride=1, w/o  shortcut conv) × 3
   
   (res4 stage, 1/16 scale)
   BottleneckBlock (c)(stride=2, with shortcut conv)
   BottleneckBlock (a)(stride=1, w/o shortcut conv) × 5
   
   (res5 stage, 1/32 scale)
   BottleneckBlock (c)(stride=2, with shortcut conv)
   BottleneckBlock (a)(stride=1, w/o  shortcut conv) × 2
   ```

   ResNet101和ResNet152的bottleneck块(a)数量较多，定义在：[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/resnet.py#L442)

   **(1) BasicStem(阀块)**[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/resnet.py#L292-L324)

   ResNet的"主干"块非常简单。它通过7×7卷积对输入图像进行两次下采样，stride=2，并通过stride=2进行最大池化max pooling。
   主干块的输出是一个特征图张量，其大小为（B，64，H / 4，W / 4）。

   * conv1 (内核大小=7，步幅=2)

   - batchnorm layer
   - ReLU
   - maxpool层(内核大小=3，跨度=2)

   **(2) BottleneckBlock**[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/resnet.py#L53-L154)

   Bottleneck块最初是在ResNet论文中提出的。该块有<u>三个卷积层</u>，其内核大小分别为1×1、3×3、1×1。3×3卷积层的输入和输出通道数小于该块的输入和输出，以提高计算效率。

   瓶颈块有三种类型，如图5所示：
     (a): stride=1, w/o shortcut conv
     (b): stride=1, with shortcut conv
     (c) : stride=2, with shortcut conv

   * **shortcut cov（used in (b), (c)）**

     ResNet有identity shortcut，增加了输入和输出的特征。对于一个阶段的第一块（*res2-res5*），使用快捷卷积层来匹配输入和输出的通道数。

   * **downsampling convolution下采样卷积，stride=2(used in (c))**

     在*res3*、*res4*和*res5*阶段的第一块，特征图由stride=2的卷积层进行下采样。由于输入通道数与输出通道数不一样，所以还使用了stride=2的快捷卷积。请注意，上面提到的\'卷积层\'包含卷积torch.nn.Conv2d和归一化（如[FrozenBatchNorm](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/layers/batch_norm.py#L14-L124)⁶）[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/layers/wrappers.py#L38-L72)
     ReLU激活在卷积和特征添加后使用（见图5）

   <img src="https://i.loli.net/2020/10/31/cAZyMvx3pfoY8gN.jpg" alt="图5.三种类型的瓶颈块。" style="zoom: 67%;" />

   ​			

2. **lateral convolution layers横向卷积层**[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/fpn.py#L65-L67)

   这一层被称为\"横向\"卷积，因为FPN最初被描绘成一个金字塔，其中主干层（sterm）被放在底部（本文中是旋转的）。横向卷积层从*res2-res5*阶段中提取不同通道号的特征，并返回256-ch特征图。

2. **output convolution layers输出卷积层**[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/fpn.py#L68-L76)

   一个输出卷积层包含3×3卷积，不改变通道数。

3. **forward process 前进过程**[代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/fpn.py#L125-L136)

   ![图6.放大到FPN原理图中涉及res4和res5的部分。放大到FPN原理图中涉及res4和res5的部分。](https://i.loli.net/2020/10/31/5xwmj4hanF1b89A.png)

   H/32，FPN的前向处理是从*res5*输出开始的(见图6)。经过横向卷积后，256通道的特征图被送入输出卷积，以P5(1/32比例)的形式登记到*结果*列表中。

   256通道的特征图也被送入上采样器（F.interpolate with nearest neighbor），并添加到res4输出中（通过横向卷积）。结果特征图经过输出卷积，结果张量P4被插入到*结果*列表中（1/16比例）。

   上面的例程（从上采样到插入到*结果*）进行了三次，最后*结果*列表中包含了*四个*时标\--即P2（1/4比例）、P3（1/8）、P4（1/16）和P5（1/32）。

4. **LastLevelMaxPool** [代码链接](https://github.com/facebookresearch/detectron2/blob/e0bffda3f503bc4caa1ae2360520db3591fd291d/detectron2/modeling/backbone/fpn.py#L165-L177)

   为了使P6输出，在ResNet的最后一个块中加入一个内核大小=1，跨度=2的最大池化层。该层只是将P5特征（1/32比例）下采样到1/64比例的特征，以便添加到*结果*列表中。



# 3. FPN好在哪？

> **自底向上**其实就是网络的前向过程。在前向过程中，feature map的大小在经过某些层后会改变，而在经过其他一些层的时候不会改变，作者将不改变feature map大小的层归为一个stage，因此每次抽取的特征都是每个stage的最后一个层输出，这样就能构成特征金字塔。<u>横向连接将high-level特征融合到low-level特征中，从而提高了低层特征的语义level</u>



# 附：Backbone Modeling的代码结构

相关文件在 detectron2/modeling/backbone 目录下。

<img src="https://i.loli.net/2020/10/31/7GbUSZedl1pxCRW.png" alt="路径" style="zoom: 67%;" />

以下是类的层次结构。

<img src="https://i.loli.net/2020/10/31/za1BidG5FHvxPUJ.png" alt="类层次结构" style="zoom:67%;" />



