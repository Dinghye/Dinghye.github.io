---
title: 【RS Detection】目标检测模型的选择（一个简单的overview）

tags:
  - 技巧
categories:
  - - 机器学习
    - 实践操作
---

针对目前的一些冠军方法和策略进行分析。

<!--more-->

> 啊写在前面：这是一个比赛的记录blog，主要还是面向遥感的目标检测竞赛进行记录和梳理的。由于之前使用的框架的缘故，本篇可能会更倾向于使用pytorch（然而在倾斜框检测很多方法都是用的tensorflow，头疼）

# 1. 相关目标检测模型

## 1.1 普通的目标检测冠军方法

​	在CV领域有很多的目标检测竞赛，下表主要展示了我搜索到的在2019-2020年的竞赛的一些冠军队伍的策略。

| Name                            | 任务类型                                                     | 模型名称                               | 策略                                                         | 链接                                                         |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CVPR/商汤2020                   | Open Images/                                                 | Faster-RCNN                            | 任务间空间自适应解耦：有效地减弱通用物体检测中分类任务和回归任务之间的潜在冲突，可以灵活插入大多检测器中。 | https://www.sohu.com/a/391981080_500659     https://arxiv.org/abs/2003.07540 |
| CVPR EPIC-Kitchens/浙江大华2020 | 厨房目标检测/                                                | Cascade-RCNN                           | 1. 骨干网：ResNet101 with FPN and  deformable convolution （DCN）；2. 数据增强：Duck filling、mix-up，显著提高所提出方法的鲁棒性    ；3. 特征增强：使用 GRE-FPN 和 Hard IoU-imbalance Sampler 提取更具代表性的全局目标特征。   ；4. 解决样本类别不平衡：Class Balance Sampling  ；5. 训练策略：随机权值平均(Stochastic Weight Averaging）    ；6. 测试策略：多尺度测试 | https://arxiv.org/abs/2006.15553                             |
| CVPR /深兰科技 2020             | 挑战赛雾天条件下的（半）监督目标检测任务                     | Cascade-RCNN                           | Baseline＝Backbone＋DCN［1］＋FPN［2］＋CascadeRCNN［3］；1．我们将Cascadercnn＋DCN＋FPN作为我们的baseline；2. 将原有head改为Double head；3．将**FFA－Net**处理过的数据集与原数据集合并训练；4．Augmentation；5．Testingtricks | https://mp.ofweek.com/ai/a445693724976                       |
| CVPR NightOwls/深兰科技 2020    | 夜间目标检测/运动模糊图像噪点、色彩信息少、数据分布笑        | Cascade-RCNN                           | Baseline = Backbone + DCN + FPN +  Cascade + anchor ratio (2.44)     ；1. 将 Cascade rcnn + DCN + FPN 作为 baseline；     2. 将原有 head 改为 Double head；     3. 将 CBNet 作为 backbone；     4. 使用 cascade rcnn COCO-Pretrained weight；    5. 数据增强；    6. 多尺度训练 + Testing tricks | https://bbs.cvmart.net/articles/3326                         |
| ECCV Google AI/百度视觉 2019    | Open Images/数据没有完全精细标注，属于弱监督任务，框选类别数目不均衡且有非常广泛的类别分布 | Cascade, Deformable, FPN,  Faster-RCNN |                                                              | https://bbs.cvmart.net/articles/664                          |

值得注意的是：

* **模型使用类别上**：基本上大家都会几乎都用到Cascade，并且未见到使用One stage的方法；反而是YOLO这种很少用到。感觉可能one stage的方法可能在<u>精度</u>上整体跟 two stage还是有差距，而竞赛中主要是看精度。此外，个人感觉two stage的这种<u>可魔改性</u>要稍微强一些。
* **竞赛主题上：**不同的竞赛，有不同的数据集和特点。特别值得注意的是CVPR 深兰科技在雾天竞赛当中，用到了[FFA除雾](https://blog.csdn.net/weixin_42096202/article/details/103277598)。这个其实有点类似于那种遥感云层，之后也许可以用上。
* **任务上：**几乎所有的这样的cv目标检测都是正框，而遥感领域更多的可能是斜框检测。中间可能还有一些不太一样的地方（比如数据朝向，提案框选择等等），也因此下面我找到一些遥感斜框的一些方法，详见1.2



## 1.2 斜框目标检测&遥感影像

> 这一部分主要讲遥感领域斜框做的公认比较好的方法和思路。但是遗憾的是，似乎很多斜框的方法框架都不开源，而且大都使用到的Tensorflow的框架，而我们以后可能都会使用pytorch……（哭

### 1.2.1 从根本出发型

针对斜框！它斜着的！这一部分提出了一些思考和方法。

1. **更科学的提案框：CVPR2019 dingjian 武大夏桂松 **

   baseline：Faster R-CNN OBB + RoI Transformer

   > 考虑到了在旋转的时候提案框的问题，更科学的提案（ROI Transformer）使用的mmdetection的框架。

   论文：[https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf)
   代码：[https://github.com/dingjiansw101/AerialDetection](https://github.com/dingjiansw101/AerialDetection)

2. **旋转不变性：ReDet CVPR 2021\武大夏桂松: 用于遥感目标检测的旋转等边检测器**

   > 提出了一个旋转等变检测器（ReDet）来解决这些问题，它明确地编码了旋转等变和旋转不变性。基于AerialDetection & mmdetection

   <img src="https://i.loli.net/2021/07/20/Cn1lT2Mo3Wyatf7.png" alt="架构" style="zoom: 10%;" />

   博客：[https://zhuanlan.zhihu.com/p/358303556](https://zhuanlan.zhihu.com/p/358303556)

   论文：[https://arxiv.org/abs/2103.07733](https://arxiv.org/abs/2103.07733)

   代码：[https://github.com/csuhan/ReDet](https://github.com/csuhan/ReDet)

3. **顺序标签点问题：Gliding Vertex 2020、RSDet** 

   > 考虑到的是顺序标签点（Sequential label points）的问题：举一个简单的例子，如果一个四边形的ground-truth是（x1,y1,x2,y2,x3,y3,x4,y4）并且所有的ground-truth并不是按一定规则顺序标注的，那么检测器有可能给出的预测结果是（x2,y2,x3,y3,x4,y4,x1,y1）。其实这两个是框是完全重合的，但是网络训练算损失的时候并不知道，它会按对应位置计算损失，此时的损失值并不为0甚至很大。

   <img src="https://pic4.zhimg.com/v2-deb5da8837c84b973a3e9c708bf0d8d7_b.jpg" alt="顺序标签问题" style="zoom: 67%;" />

   * Gliding Vertex 2020：这篇文章发现，直接就是通过**改变框的表达方式** 避免了排序的麻烦。先检测水平框，这个是没有序列问题的，然后学习水平框四个角点的偏移量来达到四边形检测的目的，其实这里的（偏移量，对应的水平框的点）配对就有排序的意思了。

     论文：https://arxiv.org/abs/1911.09358

   * RSDet：给点排序！直接粗暴

     论文：https://arxiv.org/abs/1911.08299

     野生代码by yangxue（但他妈的还是tensorflow啊，但是这是一个很好的框架）：https://github.com/yangxue0827/RotationDetection

   本部分参考：https://www.zhihu.com/search?type=content&q=%E9%81%A5%E6%84%9F%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B

### 1.2.2 从数据出发型

1. **我们是专门搞舰船的！：R3Det CVPR 2020**

   > ⚠用的是tensorflow哈。**大纵横比、密集分布和类别极不平衡的旋转物体**仍然存在挑战。在本文中，提出了一种端到端的精细单级旋转检测器，用于快速准确定位物体。特征精炼模块的关键思想是通过<u>特征插值将当前精炼的边界框位置信息重新编码为对应的特征点</u>，以实现特征重构和对齐

   论文：[https://arxiv.org/abs/1908.05612v1](https://arxiv.org/abs/1908.05612v1)

   代码：[https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation](https://github.com/DetectionTeamUCAS/RetinaNet_Tensorflow_Rotation)

2. **旋转多尺度特征：Rotation-aware and multi-scale convolutional neural network for object detection in remote sensing images 2020**

   > faster r-cnn发展而来，一种考虑了多尺度特征，不开源

   ![整体架构](https://i.loli.net/2021/07/20/2BjMATIUEHP7CWQ.png)

   论文：[https://www.sciencedirect.com/science/article/pii/S0924271620300319?via%3Dihub](https://www.sciencedirect.com/science/article/pii/S0924271620300319?via%3Dihub)

   

3. **我们的数据不太行之不平衡：YOLT 针对遥感卫星目标检测**

   > 针对「机场目标」和「其它目标」分别训练了一个检测模型，这两个检测模型的输入图像尺度也不一样，测试图像时同理，最后将不同检测模型、不同chip的检测结果合并在一起就得到最终一张卫星图像的输出。也即是说这篇文章的核心操作就是这个「不同尺度的模型融合」以及「针对机场单独训练一个模型」，这样确实是从数据出发能够很好的解决实际场景（卫星图像）中机场目标数据太少带来的问题。

   <img src="https://i.loli.net/2021/07/20/lJbBmfUc2dCzhqr.png" alt="我们遇上哪些问题啦" style="zoom: 50%;" />

   - 卫星图目标的**「尺寸，方向多样」**。卫星图是从空中拍摄的，因此角度不固定，像船、汽车的方向都可能和常规目标检测算法中的差别较大，因此检测难度大。针对这一点的解决方案是对数据做**「尺度变换，旋转等数据增强操作」**。

   - **「小目标的检测难度大」**。针对这一点解决方案有下面三点。

     a) 修改网络结构，使得YOLOV2的 stride 16,而不是32，这样有利于检测出大小在16x16→32x32

     b) 沿用YOLOV2中的passthrough layer，融合不同尺度的特征（52x52和26x26大小的特征），这种特征融合做法在目前大部分通用目标检测算法中被用来提升对小目标的检测效果。

     c) 不同尺度的检测模型融合，即Ensemble，原因是例如飞机和机场的尺度差异很大，因此采用不同尺度的输入训练检测模型，然后再融合检测结果得到最终输出。

   - **「卫星图像尺寸太大」**。解决方案有将原始图像切块，然后分别输入模型进行检测以及将不同尺度的检测模型进行融合。

   代码：https://http://github.com/CosmiQ/yolt

   论文：[https://arxiv.org/abs/1805.09512](https://arxiv.org/abs/1805.09512)

   

4. **我们的数据不太行之框框在瞎标注：CVPR 2019\旷视**

   > 案例中的基本 ground truth 边界框原本就是模糊的，导致回归函数的学习更困难：定位更精准的损失函数（KL）！

   <img src="https://i.loli.net/2021/07/20/A6qEjS3UsDdBtca.png" alt="这个框画的不行" style="zoom: 67%;" />

   链接：[https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247486607&idx=1&sn=17554987819b343e1f1bb066ea3e21ad&chksm=96f37edba184f7cdffe461edb2a89eb72bd493061cfc9b841ca5043c003a611a8855326d039d&mpshare=1&scene=22&srcid=0828kphGGoXwZThz28xhSUR7&sharer_sharetime=1567004558685&sharer_shareid=5c55e87df338791d997e8905ad2ebfe0#rd](https://mp.weixin.qq.com/s?__biz=MzIwMTE1NjQxMQ==&mid=2247486607&idx=1&sn=17554987819b343e1f1bb066ea3e21ad&chksm=96f37edba184f7cdffe461edb2a89eb72bd493061cfc9b841ca5043c003a611a8855326d039d&mpshare=1&scene=22&srcid=0828kphGGoXwZThz28xhSUR7&sharer_sharetime=1567004558685&sharer_shareid=5c55e87df338791d997e8905ad2ebfe0#rd)







# 2. 思考整个流程

1. 首先敲定主体框架

2. 数据扩增：遥感图像有一个很大的特点，就是多方向性。

   * 旋转、水平翻转、随机亮度、随机对比

   * TTA单图多测

3. 数据集制作：⚠值得注意的是！有关提交要求

4. 模型训练：⚠docker容器？

5. 问题分析及解决

