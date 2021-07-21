---
title: CHAPTER3运输层-可靠数据传输原理
comments: true
copyright: false
date: 2020-06-21 16:54:02
tags:
categories:
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gg00suosszj30m50bn0yc.jpg
toc: true
---



## 可靠数据传输原理

信息在信道传输过程中很可能会发生数据丢失。在计算机网络不同的层次有不用保证可靠传输的设计。

<!--more-->

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg00oqzpu5j30q90bd0yz.jpg)



## 可靠数据传输协议的基本结构：接口

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg00suosszj30m50bn0yc.jpg)

图中的unreliable channel就是指IP层，rdt是可靠数据传输协议的简称

**rdt_send()由应用层程序调用，把应用层数据发给可靠数据传输协议；udt_send（）由rdt调用，将数据传送给不可靠信道(IP层)； 当数据包分组到达接受方，会触发rdt_rcv() ，然后rdt就会对数据进行处理；最后rdt调用deliver_data（）向上层应用交付可靠数据**

大家注意，rdt_send()那里是单项箭头，上层网络应用只调用一次rdt_send()就不管了，剩下的工作全交给TCP,deliver_data()也是单项箭头，说明TCP把一切处理好了再交给应用层网络应用

而udr_send()和rdt_rcv()都是双向箭头，这里说明要不断交互信息来控制数据流动

## 可靠数据传输协议设计

接下来持续将近地设计rdt。大致方向如图：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg01cc5wyxj30nw05hadl.jpg)

刻画网络传输协议,可以用[**有限状态自动机**](https://www.orcode.com/question/53000_k39143.html)，用下面的图解释一下（我也不知道是啥）

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg01dfhlxjj30li06rjuf.jpg)

图中用圆圈表现当前所处状态，箭头表示状态之间的迁移，横线上方表示引起状态迁移的事件，横线下方显示活动（状态转换过程要采取的活动），每个状态之间的变迁要被准确定义