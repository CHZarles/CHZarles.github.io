---
title: CHAPTER2应用层-Email
comments: true
copyright: false
date: 2020-06-20 16:26:21
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gfzqqouthvj308c04oa9w.jpg
toc: true
---



## Email的构成组件

邮件客户端在邮件应用的外围，邮件服务器是邮件应用的核心

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyu2u1pyrj30p40bnn1c.jpg)



我们要使用邮件服务，我先要向邮件服务器申请一个账号，邮件服务器为每个用户分配一个邮箱。这个邮箱存储别人发给我们的邮件，邮件服务器有一个消息队列，储存我们要发出去的消息。邮件服务器一直是开着的，这样，邮件服务器就能确保将我们的邮件送到目的邮箱，以及确保能接收到别人发来的邮件。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyu8gjlf6j30q30ca47l.jpg)



<!--more-->



## SMTP协议

**运输层用的是TCP协议，用端口25**，采用命令响应模式

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyudbb26hj30qd0begqa.jpg)





**email是一个异步应用，发送方发送和接收方接收不需要同时，注意步骤**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyuglchcej30no05q0uw.jpg)



## SMTP交互实例

**在命令行中连接邮箱服务器就是这样交互的**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyukmfv5zj30jx0antey.jpg)