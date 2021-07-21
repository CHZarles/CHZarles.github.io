---
title: CHAPTER3运输层-RDT设计
comments: true
copyright: false
date: 2020-06-23 21:00:36
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gg00suosszj30m50bn0yc.jpg
toc: true
---

## 回顾接口，状态机

可靠数据传输协议的基本结构-接口

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg00suosszj30m50bn0yc.jpg)

状态机的表示方法

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg01dfhlxjj30li06rjuf.jpg)

**图中用圆圈表现当前所处状态，箭头表示状态之间的迁移，横线上方表示引起状态迁移的事件，横线下方显示活动（状态转换过程要采取的活动），每个状态之间的变迁要被准确定义**

<!--more-->

## RDT 1.0

**前提条件-底层信道完全可靠（理想实验）**

- 不会发生错误(bit error)
- 不会丢失分组

**因为信道完全可靠，所以发送方和接收方不需要其它的信息交互，发送方和接收方的FSM独立（没有耦合关系）**

#### 发送方的状态机：

**只有一个状态:状态机一直等着上层的调用-->上层调用rdt_send(data)**

这个事件发生后，会执行操作 packet = make_pkt(data) , udt_send(packet)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2izl2xkjj30c705k3zi.jpg)



#### 接收方状态机

**只有一个状态:等待下层调用-->下层调用rdt_rcv()**

这个事件发生后，会执行的操作 extract (packet,data)  deliver_ data(data)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2j696httj309m05bt9p.jpg)





## RDT 2.0



**前提条件-底层信道只会产生位错误（理想实验）**

**解决方法**
■利用校验和检测位错误

**如何从错误中恢复?**
■确认机制(Acknowledgements, ACK):接收方显式地告知发送方分组已正确接收
■NAK:接收方显式地告知发送方分组有错误
■发送方收到NAK后，重传分组

**基于这种重传机制的rdt协议称为ARQ(Automatic Repeat reQuest)协议**



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2jq0ig9ij30pb05eju6.jpg)



**接下来看看怎么用状态机来描述设计RDT2.0**

####  发送方的状态机:

有两个状态:

- 等待上层调用
- 等待ACK or NAK

**这样的协议我们称为停等-协议**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2juvqg6lj30cu07wq5a.jpg)



#### 接收方的状态机:

只有一个状态

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2k1h1xo5j306s0bdwgb.jpg)



## RDT 2.1 

**RDT2.1为了解决ACK出错和分组乱序/重复分组而设计出来的**

**解决思路**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2kzf9sggj30q30b945x.jpg)



**因为增加了序列号机制，状态机复杂度大大提升**



#### 发送方状态机

假设只有序列号0和序列号1

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2mcmm824j30jg0bk79r.jpg)

#### 接收方状态机

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2mvg2lysj30m70bhtew.jpg)





## RDT 2.2

目的在于精简RDT2.1

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2mywrnhdj30nz080jvb.jpg)





#### 状态机

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2n0efhbnj30lg0c2tfa.jpg)



## RDT 3.0

**RDT3.0是为了解决分组丢失而进一步设计出来的**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2ny76fhbj30p40ay447.jpg)



**发送方的状态机**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2o114bt0j30kf0bggrg.jpg)





丢包与没丢包的示意图

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2oe09fg0j30nb0b6td3.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2or8edq6j30m40bdgqt.jpg)

**接收机的状态机**

（不知道）



**RDT性能分析：瓶颈在于停等操作**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg2ozj8i51j30p40bojxw.jpg)



## 滑动窗口协议

为了打破停等协议的局限，用流水线机制发送多个分组，提高信道利用率。

下图以连续发三个分组为例，利用率比前面EDT3.0的提高了三倍

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg33pjy11uj30pk0b4win.jpg)



## 流水线协议

那些接收到但是没确认的都要缓存。看空中数据图，明显用了窗口协议的数据包密集很多。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg33u0cfwdj30pf0bfwkb.jpg)



#### 滑动窗口协议

要实现流水线机制要启用流动窗口协议:图中黑色的表示已经确认的数据,黄色的表示已经发出去的，nextseqnum表示下一个使用的序列号，蓝色表示还可以用的

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg34gs4n3cj30nx0bkn2i.jpg)