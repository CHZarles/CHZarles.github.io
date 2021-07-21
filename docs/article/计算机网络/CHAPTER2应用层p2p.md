---
title: CHAPTER2应用层p2p
comments: true
copyright: false
date: 2020-06-20 13:19:26
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gfyt4uktanj30p60ardlo.jpg
toc: true
---



## P2P概述

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyt4uktanj30p60ardlo.jpg)

## 比较cs架构和p2p架构文件分发速度



cs架构文件分发速度

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyt3oymapj30q60bnaf5.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyt89eepaj30r30cldmc.jpg)

<!--more-->

p2p的文件分发速度

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyt8z7dqbj30qs0b2afw.jpg)





## BitTorrent简单原理

BitTorrent是一种用于文件分发的流行P2P协议.

用BitTorrent 的术语来讲，参与一个特定文件分发的所有对等方的集合被称为一个洪流(torrent)。 

在一个洪流中的对等方彼此下载等长度的文件块( chunk),典型的块长度为256KB。

> 当一个对等方首次加入一个洪流时，它没有块。随着时间的流逝，它累积了越来越多的块。当它下载块时，也为其他对等方上载了多个块。一旦某对等方获得了整个文件，它也许(自私地)离开洪流，或(大公无私地)留在该洪流中并继续向其他对等方上载块。同时，任何对等方可能在任何时候仅具有块的子集就离开该洪流，并在以后重新加入该洪流中。



每个洪流具有一个基础设施结点，称为追踪器( tracker)。当一个对等方加人某洪流时，它向追踪器注册自己，并周期性地通知追踪器它仍在该洪流中。以这种方式，追踪器跟踪正参与在洪流中的对等方。

一个给定的洪流可能在任何时刻具有数以百计或数以千计的对等方。如图所示，当一个新的对等方Alice加入该洪流时，追踪器随机地从参与对等方的集合中选择对等方的一个子集( 为了具体起见，设有50个对等方)，并将这50个对等方的IP地址发送给Alice。Alice 持有对等方的这张列表，试图与该列表上的所有对等方创建并行的TCP连接。

我们称所有这样与Alice 成功地创建一个TCP连接的对等方为“邻近对等方”(在图中，Alice显示了仅有三个邻近对等方。通常，她应当有更多的对等方)。

随着时间的流逝，这些对等方中的某些可能离开，其他对等方(最初50个以外的)可能试图与Alice创建TCP连接。因此一个对等方的邻近对等方将随时间而波动。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytglvjozj30p50b67a1.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyti4hlj7j30px0bsdo0.jpg)

## 获取chunk的策略

**优先获取稀缺资源，优先传输给对自己贡献大的领近对等方，随机传输有利于打破原理的平衡**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytioak64j30pb0brtfq.jpg)



## P2P  索引方案

**索引的作用**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytlevs9nj30og0bi7bt.jpg)



### **集中性索引**

内容和文件传输是分布式的，但是内容定位是高度集中式的：单点失效问题，性能瓶颈，版权问题

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytndvj3fj30pl0bqgqs.jpg)

### **分布式索引**

**完全的分布式的架构**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcff8x75tj30vo0begwb.jpg)

**用洪泛式查询，任何收查询消息的机器都会将查询消息转发到和自己建立TCP连接的机器上，直到查到目标，再通过反向路径传回第一个查询的主机**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytssqnfyj30pj0c5agk.jpg)

### 层次索引

**结合了集中式索引和完全分布式索引**，超级节点与超级结点之间用洪泛方式查询，节点与超级节点用集中式查询

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytxsdn0xj30pj0b90xi.jpg)

**典型案例**，可自己查资料研究架构

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfytzy3jv3j30oy0ct7en.jpg)