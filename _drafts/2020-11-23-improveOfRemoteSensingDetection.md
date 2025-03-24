---
title: 【机器学习】提升模型效果的小方法

tags: 技巧

---

影响机器学习结果的因素有很多，包括模型设计、数据等等因素。本文总结了一些常见提高效果的方法

<!--more-->

# 1. 数据增强

https://zhuanlan.zhihu.com/p/41679153

我们常常会遇到数据不足的情况。比如，你遇到的一个任务，目前只有小几百的数据，然而，你知道目前现在流行的最先进的神经网络都是成千上万的图片数据。你知道有人提及大的数据集是效果好的保证。

> 如果没有很多的数据，我们怎么去获得更多的数据？

![移位网球](https://pic1.zhimg.com/80/v2-427fbfa87d132162b04e4dc8a75c446c_720w.jpg)

一个欠训练的网络会认为上述三个网球是不同、独特的。所以为了更多的数据，我们可以**对现有数据集进行微小的转变** 比如旋转（flips）、移位（translations）、旋转（rotations）等等。

一个卷积神经网络，如果能够对物体即使它放在不同的地方也能稳健的分类，就被称为具有不变性的属性。更具体的，CNN可以对移位（translation）、视角（viewpoint）、大小（size）、照明（illumination）（或者以上的组合）具有不变性。

# 2 长尾分布

https://bbs.cvmart.net/topics/1694

https://mp.weixin.qq.com/s?__biz=MzU5MTgzNzE0MA==&mid=2247487069&idx=1&sn=1011ebe0f4ec56f3e48a6d71cf50c2b1&chksm=fe29ace6c95e25f0691c53fd58796270b25b944a3bcab2d45d0608932e0324d73d7b5ec1d30f&scene=27#wechat_redirect

<img src="https://bbs.cvmart.net/uploads/images/202003/28/16/DUbs91SEKS.png?imageView2/2/w/1240/h/0" alt="长尾分布" style="zoom: 50%;" />

> CVPR2019[1]认为：当数据呈现长尾分布时，会导致分类器出现bias(势利眼)，**分类器更偏向于识别样本量充足**，类内多样性丰富的头部类，从而忽略了尾部类，这对尾部类而言是不公平的。

我们认为由于尾部ID的数量庞大，而且每个尾部ID所拥有的样本数量稀少，这会导致特征空间十分混乱，大量类别的辨识度不高，使得特征空间发生扭曲，畸变。最终网络学习得到的是一个不健康的模型，先天畸形。

常用的方法有：

## 2.1 重采样（re-sampling）相关

1. [Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR 2020](https://arxiv.org/abs/1910.09217)

   *对任何不均衡分类数据集地再平衡本质都应该只是对分类器地再均衡，而不应该用类别的分布改变特征学习时图片特征的分布，或者说图片特征的分布和类别标注的分布，本质上是不耦合的。*

   基于上述假设，有了Decoupling 以及BBN。Decoupling的核心在于图片特征的分布和类别分布其实不耦合，所以学习backbone的特征提取时不应该用类别的分布去重采样（re-sampling），而应该直接利用原始的数据分布。

   Decoupling将长尾分类模型的学习分为了两步。第一步，先不作任何再均衡，而是直接像传统的分类一样，利用原始数据学习一个分类模型（包含特征提取的backbone + 一个全连接分类器）。第二步，将第一步学习的模型中的特征提取backbone的参数固定（不再学习），然后单独接上一个分类器（可以是不同于第一步的分类器），对分类器进行class-balanced sampling学习。

2. [BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition，CVPR 2020](https://arxiv.org/abs/1912.02413)

   ![长尾分类的最佳组合来自于：利用Cross-Entropy Loss和原始数据学出来的backbone + 利用Re-sampling学出来的分类器。](https://mmbiz.qpic.cn/sz_mmbiz_png/gYUsOT36vfogQwQSEpqTKnZY1NQ1FcZbJV0icgmmUqFkTuaiafExEPLupq3ic2dpXm8xMCgJEx3d6WzxTicW8Hmetg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

3. [Dynamic Curriculum Learning for Imbalanced Data Classification，ICCV 2019](https://arxiv.org/abs/1901.06783)

   课程学习（Curriculum Learning）是一种模拟人类学习过程的训练策略，旨在从简到难。先用简单的样本学习出一个比较好的初始模型，再学习复杂样本，从而达到一个更优的解。

## 2.2 重加权（re-weighting）相关

1. [Class-Balanced Loss Based on Effective Number of Samples，CVPR 2019](https://arxiv.org/abs/1901.05555)

   这篇文章的核心理念在于，随着样本数量的增加，每个样本带来的收益是显著递减的。所以作者通过理论推导，得到了一个更优的重加权权重的设计，从而取得更好的长尾分类效果。

2. [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss，NIPS 2019](https://arxiv.org/abs/1906.07413)

3. [Rethinking Class-Balanced Methods for Long-Tailed Visual Recognition from a Domain Adaptation Perspective, CVPR 2020](https://arxiv.org/abs/2003.10780)

4. [Remix: Rebalanced Mixup,Arxiv Preprint 2020](https://arxiv.org/abs/2007.03943)

   Mixup是一个这两年常用的数据增强方法，简单来说就是对两个sample的input image和one-hot label做线性插值，得到一个新数据。实现起来看似简单，但是却非常有效，因为他自带一个很强的约束，就是样本之间的差异变化是线性的，从而优化了特征学习和分类边界

## 2.3 迁移学习（transfer learning）相关

1. [Deep Representation Learning on Long-tailed Data: A Learnable Embedding Augmentation Perspective，CVPR 2020](https://arxiv.org/abs/2002.10826)

   长尾分布中因为尾部样本缺乏，无法支撑一个较好的分类边界，这篇工作在尾部的样本周围创造了一些虚拟样本，形成一个特征区域而非原先的特征点，即特征云（feature cloud）。而如何从特征点生成特征云，则利用的头部数据的分布。图例如下

   ![图示](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfogQwQSEpqTKnZY1NQ1FcZbRTJHlSRto8GCeC8n7m4sT4XWHZzkTW1LnGPOlxzpNemDPYTcDtMPaw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

2. [Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification，ECCV 2020](https://arxiv.org/abs/2001.01536)

   作者发现在一个长尾分布的数据集中，如果我们取一个更均衡的子集来训练，其结果反而比利用完整的数据集效果更好。所以原文利用多个子集来训练更均衡的专家模型来指导一个unified学生模型。

3. [Large-Scale Long-Tailed Recognition in an Open World，CVPR 2019](https://arxiv.org/abs/1904.05160)

   该方法学习一组动态的元向量（dynamic meta-embedding）来将头部的视觉信息知识迁移给尾部类别使用。这组动态元向量之所以可以迁移视觉知识，因为他不仅结合了直接的视觉特征，同时也利用了一组关联的记忆特征（memory feature）。这组记忆特征允许尾部类别通过相似度利用相关的头部信息。

# 3. 超参数设置

比如通过对原始数据的一些检测，获得目标的大小等信息，给出更合适的Anchor大小，帮助模型进行训练识别等等。

除此之外一些其他的超参数设置可以参考：

https://blog.csdn.net/mzpmzk/article/details/80136958

# 4. 感受野

https://zhuanlan.zhihu.com/p/28492837

> 在机器视觉领域的深度神经网络中有一个概念叫做感受野，用来表示网络内部的不同位置的神经元对原图像的感受范围的大小。
>
> （在卷积神经网络中，感受野（Receptive Field）的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在输入图片上映射的区域大小。再通俗点的解释是，特征图上的一个点对应输入图上的区域）

神经元之所以无法对原始图像的所有信息进行感知，是因为在这些网络结构中普遍使用卷积层和pooling层，在层与层之间均为局部相连（通过sliding filter）。神经元感受野的值越大表示其能接触到的原始图像范围就越大，也意味着他可能蕴含更为全局、语义层次更高的特征；而值越小则表示其所包含的特征越趋向于局部和细节。因此感受野的值可以大致用来判断每一层的抽象层次。

<img src="https://pic3.zhimg.com/80/v2-5378f1dfba3e73dedafdc879bbc4c71e_720w.png" alt="感受野" style="zoom: 80%;" />

**如何计算感受野？**

https://www.cnblogs.com/objectDetect/p/5947169.html

**如何增强感受野？**

- 增加pooling层，但是会降低准确性（pooling过程中造成了信息损失）
- 增大卷积核的kernel size，但是会增加参数（卷积层的参数计算参考[[2\]](https://blog.csdn.net/dcxhun3/article/details/46878999)）
- 增加卷积层的个数，但是会面临梯度消失的问题（梯度消失参考[[3\]](https://blog.csdn.net/cppjava_/article/details/68941436)）

# 3. 优化器

https://dinghye.gitee.io/2020/11/05/DLOptimizer/

# 4. Focal Loss

https://zhuanlan.zhihu.com/p/49981234

object detection的算法主要可以分为两大类：**two-stage detector和one-stage detector**。前者是指类似Faster RCNN，RFCN这样需要region proposal的检测算法，这类算法可以达到很高的准确率，但是速度较慢。虽然可以通过减少proposal的数量或降低输入图像的分辨率等方式达到提速，但是速度并没有质的提升。后者是指类似YOLO，SSD这样不需要region proposal，直接回归的检测算法，这类算法速度很快，但是准确率不如前者。**作者提出focal loss的出发点也是希望one-stage detector可以达到two-stage detector的准确率，同时不影响原有的速度。**

> 作者认为one-stage detector的准确率不如two-stage detector原因是：**样本的类别不均衡导致的**。我们知道在object detection领域，一张图像可能生成成千上万的candidate locations，但是其中只有很少一部分是包含object的，这就带来了类别不均衡。

那么类别不均衡会带来什么后果呢？负样本数量太大，占总loss的大部分，而且多是容易分类的，因此使得优化方向与我们希望的不一样。因此，针对类别不均衡问题给，作者基于标准交叉熵损失基础上提出一种新的损失函数focal loss。

# 5. label smooth

https://zhuanlan.zhihu.com/p/76587755

> 在常见的多分类问题中，先经过softmax处理后进行交叉熵计算，原理很简单可以将计算loss理解为，为了使得网络对测试集预测的概率分布和其真实分布接近，常用的做法是使用one-hot对真实标签进行编码，作者认为这种将标签强制one-hot的方式使网络过于自信会导致**过拟合**，因此软化这种编码方式。

其方法为：label smoothing相当于减少真实样本标签的类别在计算损失函数时的权重，最终起到抑制过拟合的效果。

# 6. 非极大值抑制

https://zhuanlan.zhihu.com/p/37489043

> 目标检测的过程中在同一目标的位置上会产生大量的候选框，这些候选框相互之间可能会有重叠，此时我们需要利用非极大值抑制找到最佳的目标边界框，消除冗余的边界框。

![非极大值抑制](https://i.loli.net/2020/11/24/EJ2TRFB7irM9c4n.jpg)

其方法为：目标边界框列表及其对应的置信度得分列表，设定阈值，阈值用来删除重叠较大的边界框。**IoU**：intersection-over-union，即两个边界框的交集部分除以它们的并集。