---
title: CHAPTER3传输层-GBN协议SP协议
comments: true
copyright: false
date: 2020-06-24 09:48:25
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gg35equfh8j30pc0bbqa2.jpg
toc: true
---





## GBN（GO-BACK-N）

绿色的是已经发送的且被成功确认的分组，黄颜色表示已经发送但还没有确认，蓝色表示可用的序列号。采用累计确认机制。设置了计时器，超时重传机制。<!--more-->

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg35equfh8j30pc0bbqa2.jpg)



## GBN发送方的状态自动机

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg35rtb3plj30le0c678m.jpg)



## GBN接收方的状态自动机

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg35zavcrdj30px0bk45s.jpg)



## GBN示例

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3634u3r0j30fb0fcte0.jpg)



## GBN的缺陷

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg36q1d3jlj30ek0aidkm.jpg)

## Selective Repeat协议

SR协议是对GBN的缺陷的改进，对比GBN，多了接收方的窗口。

两个窗口并不同步

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg36yfhritj30l90e9tea.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3e6bipsoj30hr0eln0w.jpg)

SR**困境**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gg3eeafahpj30px0cytjf.jpg)