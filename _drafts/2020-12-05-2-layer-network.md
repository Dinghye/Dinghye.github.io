---
title: 【机器学习】python&numpy 两层神经网络小实验

tags:
  - 实践
categories:
  - - 机器学习
    - 实践操作
mathjax: true
---

> 这是一个简单的！用numpy以及mnist数据集的两层神经网络小实验脚本！

<!--more-->

# 1. 基本函数

## 1.1 激活函数

> 什么是激活函数？

激活函数的作用在于决定如何来激活输入信号的总和。在我们的模型中需要用到两个激活函数，第一层神经网络做线性运算的输出需要经过一个sigmoid函数，它的函数表达式为
$$
h(x)=\frac{1}{1+exp(-x)}
$$
。因为手写数字识别是一个分类问题，而softmax函数的输出是0.0-1.0之间的实数，我们可以将其解释为每个类别存在的概率，因此第二层神经网络做线性运算的输出经过一个softmax函数后输出预测结果，它的函数表达式为
$$
y_k=\frac{exp(a_k)}{\sum^{n}_{i=1}exp(a_i)}
$$
，分子是输入信号ak的指数函数，分母是所有输入信号的指数函数的和,n为所有输入信息号的数目。

```python
def sigmoid(x):
    si = []
    for i in x:
        si.append(1 / (1 + np.exp(-x)))
    return si


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x
```

## 1.2 定义损失函数

 本实验使用交叉熵误差作为损失函数，它的数学公式为
$$
E=-\sum_k t_klogy_k
$$
，log表示以e为底数的自然对数(loge)，$y_k$是神经网络的输出，$t_k$是正确解标签。

```python
def cross_entropy_error(y, t):
    delta = 1e-7  # 防止计算错误，加上一个微小值
    return -np.sum(t * np.log(y + delta))
```

## 1.3 获取权重参数的梯度

由全部参数的偏导数汇总而成的向量称为梯度，由于数值微分含有误差，所以在此处我们使用中心差分进行求导，求导公式为
$$
\frac{df(x)}{dx}=\lim_{h→0}\frac{f(x+h)-f(x-h)}{2h}
$$

```python
def numerical_gradient(f, x):
    h = 0.0001
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    count = 0
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
        it.iternext()
        count += 1
        # print(count)

    return grad
```

# 2. 构建神经网络

这一步我们将训练需要用到的两层神经网络实现为一个名为TwoLayerNet的类。参考代码中input_size,hidden_size,output_size分别表示输入层的神经元数量，隐藏层的神经元数量，输出层的神经元数量。params和grads分别为保存神经网络和梯度的字典型变量。

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, x):
        z1 = np.dot(x, self.params['W1']) + self.params['b1']
        h1 = sigmoid(z1)
        z2 = np.dot(h1, self.params['W2']) + self.params['b2']
        y = softmax(z2)
        return y

    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        z = self.forward(x)
        y = np.argmax(z, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        grads = {}
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        f = lambda W: self.loss(x, t)

        grads['W1'] = numerical_gradient(f, w1)
        grads['b1'] = numerical_gradient(f, b1)
        grads['W2'] = numerical_gradient(f, w2)
        grads['b2'] = numerical_gradient(f, b2)
        return grads

```

## 2.2 模型训练

```python
def train_model(self, x_train, t_train, x_test, t_test):
    # 定义训练循环迭代次数
    iters_num = 10000
    # 获取训练数据规模
    train_size = x_train.shape[0]
    # 定义训练批次大小
    batch_size = 10
    # 定义学习率
    learning_rate = 0.1
    
    #创建记录模型训练损失值的列表
	train_loss_list = []
	#创建记录模型在训练数据集上预测精度的列表
	train_acc_list = []
	#创建记录模型在测试数据集上预测精度的列表
	test_acc_list = []

    
    iter_per_epoch = max(train_size / batch_size, 1)
    ###请补充创建训练循环的代码
    for i in range(iters_num):
    	# 在每次训练迭代内部选择一个批次的数据
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = self.gradient(x_batch, t_batch)
    	###请补充更新模型参数的代码
        for key in ('W1', 'b1', 'W2', 'b2'):
            self.params[key] -= learning_rate * grad[key]
            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

       # 判断是否完成了一个epoch，即所有训练数据都遍历完一遍
       if i % iter_per_epoch == 0:
            ###请补充向train_acc_list列表添加当前模型对于训练集预测精度的代码
            train_acc = self.accuracy(x_train, t_train)
            ###请补充向test_acc_list列表添加当前模型对于测试集预测精度的代码
            test_acc = self.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

       # 输出一个epoch完成后模型分别在训练集和测试集上的预测精度以及损失值
       print("iteration:{} ,train acc:{}, test acc:{} ,loss:{}|".format(i, train_acc, test_acc, loss))

    # add: draw picture stuff
 	markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
```

