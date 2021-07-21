---
title: CHAPTER3传输层-UDP
comments: true
copyright: false
date: 2020-06-21 15:36:27
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gfzyuanawkj30mv0axwht.jpg
toc: true
---





## UDP协议实现的功能

1. 实现分路复用/分用
2. 增加了简单的错误校验机制（没有错误恢复）

**UDP在IP层之上没有增加什么功能，UDP将IP层的服务裸露给应用层**，而IP层的服务就是"Best effort"服务（尽力而为的服务模型），所以UDP也是"Best effort"服务模型。（可能会发生报文丢失，非按序到达）

每个UDP段的处理相互独立

UDP协议的优势

- 无连接，没有发生延时(DNS使用UDP服务)
- 头部开销小
- 容易实现
- 使用UDP，上层应用可更好地控制发送时间和发送速率

UDP协议常用于流媒体应用（容忍丢失，速率敏感）

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzyjkcv1ij30q60aptev.jpg)

其实 **UDP+合适的应用层算法** 是可以实现可靠数据传输的。



## UDP报文

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzynywopgj30cs0aa40k.jpg)

source port 和 dest port 用来实现复用/分用

checksum是校验和，用来检测是否发送错误



### 校验和工作原理

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzys2lrj9j30r00afah4.jpg)

<!--more-->

示例

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzyuanawkj30mv0axwht.jpg)












