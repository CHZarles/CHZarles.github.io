---
title: CHAPTER传输层-TCP流量控制/流量管理/拥塞控制/传输层总结
comments: true
copyright: false
date: 2020-07-02 00:04:15
tags:
categories:
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1ggcb6x61mwj30ot0b1gr5.jpg
toc: true
---

## 流量控制概览

流量控制本质上是一直速度匹配机制

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggbx4s5woqj30pe0at0y2.jpg)





空闲buffer的算法，以及处理流程。RecWindow=0时sender依然会发送小数据，避免死锁

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggbxeljmoxj30oj0auwml.jpg)

<!--more-->



## 连接管理

### TCP是建立

Seq# 是序列号的意思，初始化TCP就是分配资源和分配序列号

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggbxmrz79tj30pv0dok30.jpg)



三次握手：1.客户机发送SYN报文，SYN报文没有数据内容，任务是选择序列号

​				  2.服务器回复SYNACK ，服务器开始分配序列号和资源

​                  3.客户接受SYNACK，回复ACK,正式建立连接（可以包含数据）

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggbxs0ubuaj30fg0f3tck.jpg)



### TCP的关闭

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggby0jv8nwj30nz0bcq95.jpg)





## 拥塞控制

### 拥塞表现，定义

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggby94k14vj30oz09ztj0.jpg)



拥塞场景一：这是理想场景，c是路由器出口的带宽。λin是主机发送速率，λout是主机接受速率。throughput达到最大（来多少吐多少），分组延时太大

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggca69i18tj30nr0b8jwc.jpg)



拥塞场景二: 

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcaftcsqyj319z0lfgxd.jpg)

R是链路的带宽

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcajwtuk5j31e90n04fh.jpg)

拥塞场景三：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcaku2xvrj31d60lx7kj.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcangw04aj31g00llwsj.jpg)



### 拥塞控制的具体方法

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcaqht63uj30p507lwkb.jpg)





#### 网络辅助拥塞控制：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcasrg8j7j30po087jwt.jpg)



#### TCP的拥塞控制：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcazd252rj30pl09mtfs.jpg)



##### AIMD详解

MSS:最大段长度

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcb1ec4tlj30o40bm0xu.jpg)



##### SS慢启动

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcb448bnzj30p00c5wk8.jpg)





##### SS和AIMD之间的转换时机SS

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcb6x61mwj30ot0b1gr5.jpg)

蓝线是tcp早期版本，congwin直接降到了1，黑线是新版本，只降到threshold。loss事件是指检测到拥塞（前面有讲）。但不同的loss事件TCP有不同的处理方案。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcbb5gzioj30q509p0xk.jpg)



代码算法表示

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcbg87jkgj30f80cdn38.jpg)

##### 拥塞控制总结

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcbbl8xybj30pc0a1wky.jpg)



### 例题：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcbj4ntckj30pl07vdo3.jpg)





传输层总结

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ggcboekeqaj30oq0ay0x6.jpg)