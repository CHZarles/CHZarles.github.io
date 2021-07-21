---
title: CHAPTER3传输层多路复用多路分用
comments: true
copyright: false
date: 2020-06-21 14:59:56
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gfzxd7xjimj30ql0bzak8.jpg
toc: true
---



## 多路复用多路分用

**多路复用多路分用技术是计算机网络层传输层的必要功能**，传输层为不同应用程序提供逻辑通讯服务，而一个主机会同时有多个进程通信，所以传输层必须实现多路复用和多路分用。

**如果某层的一个协议对应直接上层的多个协议/实体，则需要复用/分用（看下图解释）**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzxd7xjimj30ql0bzak8.jpg)

<!--more-->

### 分用的具体工作

IP数据报包裹着传输层的内容

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzxirlrztj30q10acahd.jpg)

#### 面向无连接的分用

1. 利用端口号创建socket
2. UDP的socket用二元组标识(目的IP地址,目的端口号)
3. 当目的主机收到传输层的报文，就会根据端口号送到对应的应用



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzxq263cej30pp07a0yw.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzxsnurw3j30nx0bmn2j.jpg)

这里的DP是目的端口号，SP是源端口号

#### 面向连接的分用

1. TCP的socket用四元组来标识
2. 目的服务器可能会为不同的应用开不同的socket

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzxuw54hdj30qr08443z.jpg)

**TCP协议是一对一的，一个客户机进程对应一个服务器进程**。所以TCP要用更多的信息来标识。如果像UDP那样仅用(IP,PORT)来标识socket，就无法区分下面p2-p6 和 p3-p5 两个连接

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzy0k6ssjj30of09htcm.jpg)



现在通常是让一个进程创建多个线程来维持多个TCP连接，WEB服务器就是很好的例子。图中的中间的机器只有一个进程，但是同时维持了不同的TCP连接

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzy3rfsp4j30qs0ajjvf.jpg)