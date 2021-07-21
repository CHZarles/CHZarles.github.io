---
title: 人工神经网络导读
comments: true
copyright: false
date: 2019-10-21 23:49:46
tags: [数学建模]
categories: [随笔]
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## 人工神经网络

先看人工神经网络的Wiki定义：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g869h2wqzaj30nl04kdhh.jpg)

简单地说就是，人工神网络是一个学习模型，这个模型有着类似人的神经网络的反馈机制和联系机制，通过输入数据来训练这个模型，可以使得这个模型能够模拟对应的函数（不管这个函数是怎么样的函数，线性还是非线性）。由于这个模拟是基于输入样本的，所以样本的质量很大程度影响模型的性能。



## 人工神经元模型

人工神经元模型是人工神经网络的一个基本概念，先看一张图：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86nqwpr25j307x05ywf6.jpg)
<!--more-->

> 引用一段话，来解释这个图：
>  图中X1到Xn是从其它神经元传入的输入信号，Wi1到Win分别是传入信号的权重，**θ**表示一个阈值，或称为**偏置（bias）**，偏置的设置是为了正确分类样本（？？？），是模型中一个重要的参数。神经元综合的输入信号和偏置（符号为-1~1）相加之后产生当前神经元最终的处理信号net，该信号称为**净激活或净激励**（net activation），激活信号作为上图中圆圈的右半部分f（*）函数的输入，即f(net)； f称为**激活函数或激励函数**（Activation Function），**激活函数的主要作用是加入非线性因素，解决线性模型的表达、分类能力不足的问题**。上图中y是当前神经元的输出。 

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86o20n587j30ez03zjrz.jpg)

上图是加权和（net）的计算方法，也可写成矩阵形式。



## 常用的激活函数

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86o62gm5hj30oc0at41k.jpg)

**注意上图的两个导数**，下面是部分函数讲解，**注意，下面大部分内容来自:**[人工神经元模型](https://www.cnblogs.com/mtcnn/p/9411849.html)

### Sigmoid函数

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vk9vojkj30go0b4dg1.jpg)

Sigmoid函数的特点是会把输出限定在0~1之间，如果是非常大的负数，输出就是0，如果是非常大的正数，输出就是1，这样使得数据在传递过程中不容易发散。

Sigmod有两个主要缺点，一是Sigmoid容易过饱和，丢失梯度。从Sigmoid的示意图上可以看到，神经元的活跃度在0和1处饱和，梯度接近于0，这样在反向传播时，很容易出现梯度消失的情况，导致训练无法完整；二是Sigmoid的输出均值不是0，基于这两个缺点，SIgmoid使用越来越少了。



###  tanh函数



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vkmp5cuj30a006hq2w.jpg)



tanh是Sigmoid函数的变形，tanh的均值（就是平均值）是0，在实际应用中有比Sigmoid更好的效果。



###  ReLU函数

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vl4mpsuj308n05uq37.jpg)



ReLU是近来比较流行的激活函数，当输入信号小于0时，输出为0；当输入信号大于0时，输出等于输入。



**ReLU的优点：**

-  ReLU是部分线性的，并且不会出现过饱和的现象，使用ReLU得到的随机梯度下降法（SGD）的收敛速度比Sigmodi和tanh都快。
- . ReLU只需要一个阈值就可以得到激活值，不需要像Sigmoid一样需要复杂的指数运算。

**ReLU的缺点：**

在训练的过程中，ReLU神经元比价脆弱容易失去作用。例如当ReLU神经元接收到一个非常大的的梯度数据流之后，这个神经元有可能再也不会对任何输入的数据有反映了，所以在训练的时候要设置一个较小的合适的学习率参数。



### Leaky-ReLU

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vlfnr51j30gk0agmyp.jpg)



相比ReLU，Leaky-ReLU在输入为负数时引入了一个很小的常数，如0.01，这个小的常数修正了数据分布，保留了一些负轴的值，在Leaky-ReLU中，这个常数通常需要通过先验知识手动赋值。



###  Maxout（表示看不懂，先留个坑）

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vmjwe61j30mh08jdld.jpg)



Maxout是在2013年才提出的，是一种激发函数形式，一般情况下如果采用Sigmoid函数的话，在前向传播过程中，隐含层节点的输出表达式为：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vmw39gsj307m0160sv.jpg)

其中W一般是二维的，这里表示取出的是第i列，下标i前的省略号表示对应第i列中的所有行。而在Maxout激发函数中，在每一个隐含层和输入层之间又隐式的添加了一个“隐含层”，这个“隐隐含层”的激活函数是按常规的Sigmoid函数来计算的，而Maxout神经元的激活函数是取得所有这些“隐隐含层”中的最大值，如上图所示。

Maxout的激活函数表示为：
$$
   f( x )=Max(wT1x+b1,wT2x+b2)
$$
可以看到，ReLU 和 Leaky ReLU 都是它的一个变形（比如，*w*1,*b*1=0 的时候，就是 ReLU）。

Maxout的拟合能力是非常强的，它可以拟合任意的的凸函数，优点是计算简单，不会过饱和，同时又没有ReLU的缺点（容易死掉），但Maxout的缺点是过程参数相当于多了一倍。





## 神经网络的分类

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86oyykq85j30ig07y3zn.jpg)

神经网络由大量的神经元互相连接而构成，根据神经元的链接方式，神经网络可以分为3大类。



###  前馈神经网络(　Feedforward Neural Networks )



前馈网络也称前向网络。这种网络只在训练过程会有反馈信号，而在分类过程中数据只能向前传送，直到到达输出层，层间没有向后的反馈信号，因此被称为前馈网络。前馈网络一般不考虑输出与输入在时间上的滞后效应，只表达输出与输入的映射关系；

**感知机( perceptron)**与**BP神经网络**就属于前馈网络。下图是一个3层的前馈神经网络，其中第一层是输入单元，第二层称为隐含层，第三层称为输出层（输入单元不是神经元，因此图中有2层神经元）。



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vnnlx5nj30eg0a8abh.jpg)



### 反馈神经网络　(　Feedback Neural Networks )



反馈型神经网络是一种从输出到输入具有反馈连接的神经网络，其结构比前馈网络要复杂得多。反馈神经网络的“反馈”体现在当前的（分类）结果会作为一个输入，影响到下一次的（分类）结果，即当前的（分类）结果是受到先前所有的（分类）结果的影响的。

典型的反馈型神经网络有：Elman网络和Hopfield网络。



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vnz65yrj30fq0a0jst.jpg)



### 自组织网络 ( SOM ,Self-Organizing Neural Networks )



自组织神经网络是一种无导师学习网络。它通过自动寻找样本中的内在规律和本质属性，自组织、自适应地改变网络参数与结构。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86voc0ewyj30k608saab.jpg)



## BP神经网络

### 简介：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vulow3ij30n007zdio.jpg)

对应起来就是：

- **BP ，即backpropagation**，是一种常规的神经网络训练方法。
- 这是有导师学习神经网络。可概括为"[delta rule]( https://en.wikipedia.org/wiki/Delta_rule )"（没看懂） ,但要事先知道怎么计算误差，计算输出，以及怎么样的输出是合理的。
- 激活函数必须可导

### 学习步骤

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vt3kpx0j30na080770.jpg)

对应起来就是：

- 传播数据：
  - 正向传播：不断输入输入样本，计算net，调用激活函数，将计算结果层层传递，直到输出最终结果。
  - 反向传播：如果某次输出和预计结果不符合，将误差反馈，误差也是以传递地形式返回到输出来，修正输入/权值。
- 权值修正：要沿梯度下降最快方向修正

#### 向前传递的过程：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vphj10mj30ix0aeq7e.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86qa8cus9j30l60aejwg.jpg)

并不是所有误差的计算方法都是Z-Y，图中只是如此表示误差而已。



### 误差反馈（反向传播）

续上图：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86qf74jdgj30lb0aegrb.jpg)

**细看红框里的计算，这一步是开始修正权值的计算（因为所有的f(*)的误差都反馈完了）。这里修正方法叫梯度下降法**
$$
新的权值 = 原权值 + 误差项 = 原权值 + η(学习率)*δ(反馈的误差)*激活函数微分项*Xi
$$
![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86vqjms9ej30mp0ab0y1.jpg)



## 数字归一化

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g86qr721zbj30nf09sjtt.jpg)

------

