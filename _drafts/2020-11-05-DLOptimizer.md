---
title: 【深度学习】深入浅出深度学习优化器
tags:
  - 优化器
categories:
  - - 机器学习
    - 原理
mathjax: true

---


（唉，简介好难写啊）是一个讲深度学习优化器原理和笔记的东西，你将能看到：1. 什么是梯度和梯度下降？ 2. 什么是优化器？解决的问题是什么？ 3. 在深度学习中优化器有哪些困难？又是怎么克服的？ 以及 4. 常见的各种优化器的优缺点和原理

<!--more-->

# 0. 梯度和梯度下降

1. 什么是梯度？

   在微积分里面，对多元函数的参数求∂ 偏导数，把求得的各个参数的**偏导数以向量的形式写出来**，就是梯度。比如函数f(x,y), 分别对x,y求偏导数，求得的梯度向量就是(∂f/∂x, ∂f/∂y)T,简称grad f(x,y)或者▽f(x,y)。对于在点(x0,y0)的具体梯度向量就是(∂f/∂x0, ∂f/∂y0)T.或者▽f(x0,y0)，如果是3个参数的向量梯度，就是(∂f/∂x, ∂f/∂y，∂f/∂z)T,以此类推。

2. 什么是梯度下降？

   梯度向量从几何意义上讲，就是**函数变化增加最快的地方**。沿着梯度向量的方向，<u>可以找到函数的最大值或者局部最大</u>值。同理，沿着梯度的反方向，梯度减少最快，可以找到局部最小值或者全局最小值。

   

# 1. 深度学习与优化器

> **什么是优化器？**

​		深度学习可以归结为一个优化问题（一般还是一种非凸优化）最小化目标函数$J(\theta)$  ；最优化的求解过程，首先求解目标函数的梯度$ \bigtriangledown J(\theta) $ ，然后将参数$\theta $向负梯度方向更新，$\theta_t= \theta_{t-1} -\eta \bigtriangledown J(\theta) $  , $\eta$ 为学习率，表明梯度更新的步伐大小。

​		最优化的过程依赖的算法**称为优化器**，可以看出深度学习优化器的两个核心是**梯度**与学习率，前者决定参数<u>更新的方向</u>后者决定参数<u>更新程度</u>。深度学习优化器之所以采用梯度是因为，对于高维的函数其更高阶导的计算复杂度大，应用到深度学习的优化中不实际。故本篇主要就深度学习中常用到的<u>梯度优化器</u>进行讲解。



> *凸优化和非凸优化？

​		非凸优化是和凸优化对应的，凸优化就像比如二次函数，它的最优解是可以公式的出来的，书面化一点的语言就是，**任何局部最优解就是全局最优解**（下图左）。对于这种情况，贪婪算法或者梯度下降法都可以收敛到全局最优解。

<img src="https://i.loli.net/2020/11/05/CHMFxmQaWBbEhrp.jpg" alt="图1. 凸优化和非凸优化" style="zoom: 50%;" />

​		然而非凸优化问题则可能存在无数个局部最优点，同时对于这种情况下还有容易出现一种”**鞍点**“，就是在某一些方向梯度下降，另一些方向梯度上升，形状似马鞍，<u>但是这一点的导数也为0，从某些维度看是极小值，另一些维度看又是极大值</u>。这就很令人讨厌了！如下图**黑点所示**。

<img src="https://i.loli.net/2020/11/05/NSmjn2hkfTbXqr1.jpg" alt="图2. 鞍点" style="zoom: 80%;" />





# 2. 基本的 BGD、SGD、MBGD

这三者都是梯度下降的最常见的变形，区别在于我们用**多少数据**来计算目标函数的梯度。

## 2.1 批量梯度下降BGD

首先是最基础的Batch Gradient Descent，采用<u>整个训练集的数据</u>来计算cost function对参数的梯度
$$
\theta = \theta -\eta · \bigtriangledown_\theta J(\theta )
$$
这种方法对于凸函数来说可以收敛到全局极小值，对于非凸函数可以收敛到<u>局部极小值</u>。但是由于这种方法是在一次更新当中，对于整个数据集计算梯度，所以计算起来**很慢**，遇到大量的数据集很麻烦，并且不能投入新数据实时更新模型。

```python
for i in range(nb_epochs):
	params_grad = evaluate_gradient(loss_funtion,data,params)
	params = params -learning_rate * params_grad
```

## 2.2 随机梯度下降 SGD

相对于BGD，Stochastic Gradient Descent提出了更加科学的解决方案。SGD每次更新是先抽出一个随机样本$J(x_i)$，并对**其中的每个样本进行梯度更新**。
$$
\theta = \theta -\eta ·\bigtriangledown_\theta J(\theta;x^{(i)};y^{(i)})
$$

```python
for i in range(nb_epochs):
	np.random.shuffle(data)      # 随机采一个样本J(xi)来更新参数 
	for example in data:         # SGD对于每一个样本进行梯度更新
		params_grad = evaluate_gradient(loss_function,example,params)
		params = params - learning_rate * params_grad
```

这个方法的核心思想有点像*爬山算法*那样，选取部分样本，以达到$\theta$的最优解。同时对于大数据集来说，可能会有相似的样本，使用BGD会在最后的计算产生冗余。而SGD一次只进行一次更新，就<u>没有了冗余，且速度快，可新增样本。</u>除此之外，由于其随机性，相对于BGD，它有可能<u>跳到更好的局部极小值处</u>。

但是同样，这种方法也会造成模型的**不稳定性**：SGD的噪音相对于BGD的要多，使得它的每一次迭代并不是都向着整体最优方向；且因为SGD更新频繁，造成其cost function有严重震荡；而且其准确度受到影响，结果并不是全局最优。

<img src="https://i.loli.net/2020/11/05/hGdeKEZNv1bSm6g.png" alt="图3. SGD梯度下降，可以看到这个过程是比较曲折的" style="zoom:50%;" />

## 2.3 小批量梯度下降法 MBGD

有没有一个折中的办法呢？Mini-Batch Gradient Descent小批量梯度下降法可能是一种选择。

MBGD 每一次利用**一小批样本**，即 n 个样本进行计算，这样它可以降低参数更新时的方差，收敛更稳定，另一方面可以充分地利用深度学习库中高度优化的矩阵操作来进行更有效的梯度计算。
$$
\theta = \theta - \eta · \bigtriangledown J(\theta;x^{(i:i+n)};y^{(i:i+n)})
$$

```python
for i in range(nb_epochs):
	np.random.shuffle(data)
	for batch in get_batches(data, batch_size=50):
		params_grad = evaluate_gradient(loss_function, batch, params)
		params = params - learning_rate * params_grad
```

其中涉及到一”批“的定义问题，多少样本算是一批呢？一般来说这个超参数取值在50~256之间。

## 2.4 存在的问题

Mini-Batch Gradient Descent 看起来似乎已经是一个非常好的方法了，但是仍然存在一些问题（似乎是这样简单梯度下降的通病）：

* **收敛性问题**：它不能保证很好的收敛性。learning rate太小，收敛速度很慢，太大，loss function就会在极小值处不停的震荡甚至偏离。对于非凸函数，还要避免陷于局部极小值处，或者鞍点处，因为鞍点周围的error是一样的，
* 所有维度的梯度都接近于0，SGD 很容易被困在这里。**会在鞍点或者局部最小点震荡跳动，因为在此点处，如果是训练集全集带入即BGD，则优化会停止不动，如果是mini-batch或者SGD，每次找到的梯度都是不同的，就会发生震荡，来回跳动。**

* 此外，这种方法是对所有参数更新时应用**同样的learning rate**，如果我们的数据是稀疏的，我们更希望对**出现频率低的特征进行大一点的更新**。（比如下图出现的，当数据出现长尾分布的时候，则会造成模型对头部或者尾部类别数据的效果很差）

  <img src="https://i.loli.net/2020/11/05/AOhie3wUEXvuHBk.png" alt="长尾数据分布下的SGD" style="zoom: 40%;" />



# 3. 梯度下降的改进

为了解决上述问题，一些研究在基础的梯度优化的基础上进行了提升。这一部分主要介绍几个典型的优化器：

## 3.1 针对于收敛性——动量优化派

### 3.1.1 Momentum 动量法

在前面的基础梯度下降算法中我们提到，决定方向的就是当前位置的梯度。但是**刚开始的时候梯度是不稳定的，方向改变是很正常的**，梯度就是抽疯了似的一下正一下反，导致做了很多无用的迭代。而动量法做的很简单，**相信之前的梯度**。如果梯度方向不变，就越发更新的快，反之减弱当前梯度。
$$
v_t = \gamma v_{t-1}+\eta \bigtriangledown_\theta J(\theta)
$$

$$
\theta = \theta - v_t
$$

其中$\gamma$是一个超参数，一般设置为0.9左右。这种方法就相当于给了一个加速度，让运动求解具有一个质量（惯性）。

<img src="https://i.loli.net/2020/11/05/cJVeEXpQBm9z7uU.png" alt="图4. 动量图示" style="zoom:40%;" />

而在实际情况中就变成这样子：

<img src="https://i.loli.net/2020/11/05/tZmO3wpSjko6dKA.png" alt="图5. 有无动量加入的下降过程对比" style="zoom: 67%;" />

这样一来，梯度方向不变的维度上速度变快，梯度方向有所改变的维度上的更新速度变慢，这样就可以加快收敛并减小震荡。

### 3.1.2 牛顿加速梯度 NAG

牛顿加速梯度（NAG, Nesterov accelerated gradient）算法，是Momentum动量算法的变种。更新模型参数表达式如下：
$$
v_t =\gamma v_{t-1}+\eta \bigtriangledown_\theta J(\theta - \gamma v_{t-1})
$$

$$
\theta = \theta - v_t
$$

其中$\gamma$是一个超参数，一般也设置为0.9左右。这种方法与Momentum 的差别在于：它用$\theta- \gamma v_{t-1}$的值来近似当作参数下一步会变成的值，意思是计算在未来的位置（对未来的预测）。

<img src="https://i.loli.net/2020/11/05/7H5yEp1UXk8cKJS.png" alt="图6. NAG示意图" style="zoom: 50%;" />

为什么要计算未来位置而不是当前位置的梯度呢？在前面的Momentum方法中，相当于是小球从山上滚下来是盲目的沿着坡滚。而NAG相当于是给了小球一双眼睛，让小球看清楚自己所在的地方的情况，当前方碰上上坡时，能够及时减速，从而保证它的适应性。

![图7. NAG优化原理解释](https://i.loli.net/2020/11/05/OTb3i8qUXc2jBAd.png)

蓝色是 Momentum 的过程，会先计算当前的梯度，然后在更新后的累积梯度后会有一个大的跳跃。 而 NAG 会先在前一步的累积梯度上(brown vector)有一个大的跳跃，然后衡量一下梯度做一下修正(red vector)，这种预期的更新可以避免我们走的太快。NAG 可以使 RNN（循环神经网络，用于处理序列问题）在很多任务上有更好的表现。

目前为止，我们可以做到，**在更新梯度时顺应 loss function 的梯度来调整速度，并且对 SGD 进行加速**。



## 3.2 针对于参数——参数更新派

前面的”动量“思想的引入，让我们更好的能把握前进的方向，避免了大幅度的震荡。而这一部分我们主要思考，我们希望模型能够根据参数的重要性而对不同的参数进行不同程度的更新。

### 3.2.1 自适应梯度算法（Adagrad）


$$
\theta_{t+1,i} = \theta_{t,i}-\frac{\eta}{\sqrt{G_{t,ii}+\epsilon }}·g_{t,i}
$$
其中，$g$为t时刻参数$\theta_i$ 的梯度$g_{t,i}=\bigtriangledown_{\theta}J(\theta_i)$。 

而$G_t$是一个对角矩阵，$ii$是t实可参数$\theta_i$的梯度平方和。通过Gt的加入，使得学习率能够自动进行调节。一般来说超参数$\eta$选取0.01

这个算法就**可以对低频的参数做较大的更新**，**对高频的做较小的更新**，也因此，**对于稀疏的数据它的表现很好，很好地提高了 SGD 的鲁棒性**，例如识别 Youtube 视频里面的猫，训练 GloVe word embeddings，因为它们都是需要在低频的特征上有更大的更新。

但是它的缺点是分布会不断累积，使得学习率最终会收缩变得非常小。

### 3.2.2 Adadelta & RMSprop

两者都是在Adagrad 的基础上提出为了解决学习率急剧下降问题的。

1. Adadelta

   有了Adagrad的基础思路（关联梯度与学习率实现自适应），Adadelta进行了新的假设，它将原来的G换成了梯度平方的衰减平均值——指数衰减平均值，以此来解决Adagrad的学习率急剧下降问题。

   其中：
   $$
   E[g^2]_t = \gamma E[g^2]_{t-1}+(1-\gamma)g^2_t
   $$
   其中γ一般设定为0.9

2. RMSprop

   类似的，RMSprop 与 Adadelta 的第一种形式相同：（使用的是指数加权平均，旨在消除梯度下降中的摆动，与Momentum的效果一样，某一维度的导数比较大，则指数加权平均就大，某一维度的导数比较小，则其指数加权平均就小，这样就保证了各维度导数都在一个量级，进而减少了摆动。允许使用一个更大的学习率η）

## 3.3 综合且实用的：Adam：Adaptive Moment Estimation

（终于……快要写完了……

这个算法是另一种计算每个参数的自适应学习率的方法。**相当于 RMSprop + Momentum**

除了像 Adadelta 和 RMSprop 一样存储了过去梯度的平方 vt 的指数衰减平均值 ，也像 momentum 一样保持了过去梯度 mt 的**指数衰减平均值**：
$$
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t = \beta_2v_{t-1}+(1-\beta_2)g^2_t
$$

如果 mt 和 vt 被初始化为 0 向量，那它们就会向 0 偏置，所以做了**偏差校正**，通过计算偏差校正后的 mt 和 vt 来抵消这些偏差：
$$
\hat{m}_t =\frac{m_t}{1-\beta^t_2}
$$

$$
\hat v_t= \frac{v_t}{1-\beta^t_2}
$$

即有梯度更新规则
$$
\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{\hat v_t}+\epsilon }\hat m_t
$$
超参数设定值: 建议 β1 ＝ 0.9，β2 ＝ 0.999，ϵ ＝ 10e−8

# 4. 效果

![图8. 几种优化器效果比较](https://i.loli.net/2020/11/05/LSaMpBjf2xw6bo1.gif)

# 5. 总结

一般来说，实际操作中Adam 是最好的选择。本文主要梳理了关于优化器的原理和其发展思路。

一些地方将优化器分成”可以自适应参数“和”不可以自适应参数“两类，有些道理，但是感觉好像又不太恰当。实际上这些优化器是针对基本的梯度下降中出现的两个大问题进行发展的：

* 其一为收敛性的问题，基础的梯度优化不够稳定，且容易陷入局部最优，效率低
* 其二为学习率死板，而对于稀疏数据学习效果较差

针对这两个问题，在基础的梯度下降下，发展出了针对于收敛性的Momentum及其变形，以及针对于参数的Adagrad及其变形，最终综合考虑结合出了效果较好的Adam。（所以这么讲似乎是一个基础类，两个发展的方向，最终综合到Adam这么个结构）



# *参考*

1. 从0开始机器学习：https://lizhyun.github.io/2019/07/08/从零开始的机器学习-一/ 
2. 【AI初识境】为了围剿SGD大家这些年想过的那十几招(从momentum到Adabound)：https://zhuanlan.zhihu.com/p/57860231
3. 机器之心-优化器：https://www.jiqizhixin.com/graph/technologies/fa50298e-1a85-4af0-ae96-a82708f4b610
4. 深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）：https://cloud.tencent.com/developer/article/1118673
5. 机器学习：各种优化器Optimizer的总结与比较：https://blog.csdn.net/weixin_40170902/article/details/80092628
6. 深度学习优化器总结：https://zhuanlan.zhihu.com/p/58236906

> **更多阅读：**

[如何理解梯度下降](https://zhuanlan.zhihu.com/p/28060786)

[为什么我们更宠爱”随机“梯度下降SGD](https://zhuanlan.zhihu.com/p/28060786)

[为什么随机梯度下降方法能够收敛？](https://www.zhihu.com/question/27012077/answer/122359602)