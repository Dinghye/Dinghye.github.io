---
title: 【计算机操作】Linux实用指北

tags: 
- 技巧
categories:
- - 机器学习
  - 实践操作
---

一些常用到的小方法命令合集

<!--more-->

# 1. 批量文件格式转换

利用shell中的for循环，以及convert，首先需要安装imagemagick

```shell
sudo apt-get install imagemagick
```

然后进行操作，值得注意的是“$i”以及"${i%.*}"的使用

```shell
for i in *.png ; do convert "$i" "${i%.*}.jpg" ; done
```

用过之后可以相同的rm掉转换之前的文件



# 2. cuda run out of memory

问题出现在机器学习训练代码续跑的时候。

1. 方案1：关闭不需要的进程

   可以先用 nvidia-smi查看一下当前cuda的所有进程，在讲对应过分的进程关掉kill -9 -pid

2. 方案2：减少bach

   额就权衡一下，试跑一下

但是，在续跑的时候，无论怎么关闭都无法解决……



# 3. nodejs/npm升级

有的时候会莫名其妙就装了奇奇怪怪版本的js……可以选择进行更新。由于npm好像是属于nodejs的一部分，更新nodejs的时候会自动更新npm（如果自己直接更新npm很有可能造成nodejs和npm版本不匹配的问题）

首先可以查看一下两者的版本号`nodejs -v`,`npm -v`

![版本号查看](https://i.loli.net/2020/11/01/nQi52JyubjZ6BzU.png)

然后需要使用npm全局安装一个管理node版本的管理模板n

```shell
npm i -g n   #或者在后面加--force 
```

过程中有可能需要使用root权限，详细见4如何获得root权限

然后升级node版本

```shell
n latest # 升级到最新版本（最近稳定版本换latest为stable）
```

退出后进入查看node和npm版本即可，建议使用latest（我使用的stable好像对应的npm版本还是很低，有的还不是很支持……根据需要来安装吧）

> 参考：https://blog.csdn.net/guzhao593/article/details/81712016



# 4. root权限获得

```shell
sudo su root 
```

