---
title: CHARPTER3传输层-TCP概述+可靠传输原理
comments: true
copyright: false
date: 2020-06-24 17:02:15
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gg3ho3fj90j30q10bnwno.jpg
toc: true
---



## TCP概述

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3ho3fj90j30q10bnwno.jpg)





## TCP的段结构

<!--more-->

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3hxd8ukrj30kp0c1465.jpg)



TCP既不是完全GBN也不是百分百的SR

小demo(据说这图有错)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3i40afa1j30pz0bn7bu.jpg)





## TCP实现可靠数据传输

概述列表

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3i5ijpnmj30ml08pgqo.jpg)

流水线机制为了保证效率，累计确认机制保证传输可靠，定时器机制应当报文丢失。

#### 解决计时问题

**应该怎么合理设置超时时间?(主要问题是算RTT)**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3i9wsrucj30q10bmwm4.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggbx1bw6ajj30wk0blqc7.jpg)



## TCP发送方执行的事件

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3idf4dyej30px096gr1.jpg)

核心伪代码

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3ieo1ypxj30dd0bmaei.jpg)

**重传示例**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3isng3vsj30qr0bp0yw.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3iwke9vyj30aj0dxgot.jpg)

## TCP接收方执行的动作

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3jfv39byj30io0avwmd.jpg)

快速重传机制

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3k24ul2tj30oy0bmjzq.jpg)

伪代码

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3k33wjicj30j80don2l.jpg)