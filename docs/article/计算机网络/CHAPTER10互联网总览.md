---
title: Chapter1-互联网总览
comments: true
copyright: false
date: 2020-05-18 20:15:57
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gewve9iow5j30s10g6afb.jpg
toc: true
---





## 前言

本系列博文**内容全部摘抄自第六版《计算机网络-自顶向下的方法》**

属于**完全非原创系列**

**一个听不进课的辣鸡的自救系列**

**因为第六版已经是十年前的版本了，所以有的知识不是最新的**

[点击获取资源](https://feater.top/exam/29/)



## Chapter1 内容总览

***overview*:**

- **what’s the Internet?**
- **what’s a protocol?**
- **network edge; hosts, access net, physical media**
- **network core: packet/circuit switching, Internet structure**
- **performance: loss, delay, throughput**
- **security**
- **protocol layers, service models**
- **history**



## 从 '组件' 的角度总览互联网

**组成互联网的元素总览**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gewve9iow5j30s10g6afb.jpg)

<!--more-->

### **Compute Device**

 hosts（主机），end-system（端系统） ，两者大概是同样的意思。

### **Communication links（通信链路）**

就是将设备连接起来的线，有光纤,铜线,无线电,卫星，它们有个参数叫频宽(以后介绍)。

### **Packet switches（交换机）**

封包转运工具，即router(路由器)，路由器会根据你封包的目的地为你选择最佳路径。路由器之间是存在信息交流的。



**端系统** **通过通信链路( communication link) 和 分组交换机( packet switch)连接到一起**。通信链路不同类型的物理媒体组成。这些物理媒体包括同轴电缆、铜线、光纤和无线电频谱。当一台端系统要向另-台端系统发送数据时，发送端系统将数据分段，并为每段加上首部字节。由此形成的信息包用计算机网络的术语来说称为分组( packet)。这些分组通过网络发送到目的端系统，在那里被装配成初始数据。

分组交换机从它的一条 **入通信链路** 接收到达的分组，并从它的一条 **出通信链路** 转发该分组。市面上流行着各种类型、各具特色的分组交换机，但在当今的因特网中，两种最著名的类型是**路由器(router)** **和链路层交换机( link - layer switch)。**这两种类型的交换机朝着最终目的地转发分组。**链路层交换机通常用于接入网中**，而**路由器通常用于网络核心中。**从发送端系统到接收端系统，**一个分组**所经历的一系列通信链路和分组交换机称为通过该网络的**路径(route或path)。**



### 因特网服务提供商(  ISP)

**端系统** **通过因特网服务提供商( Internet Service Provider, ISP) 接入因特网**，包括 如本地电缆或电话公司那样的住宅区ISP、公司ISP、大学ISP,以及那些在机场、旅馆、咖啡店和其他公共场所提供WiFi接入的ISP。



**每个ISP是一个由多个分组交换机和多段通信链路组成的网络（看上面的图）。**各ISP为**端系统提供了各种不同类型的网络接入**，包括如 线缆调制解调器 或 DSL 那样的住宅宽带接人、高速局域网接入、无线接入 和 56kbps拨号调制解调器接入 。**ISP  也为内容提供者提供因特网接入服务**，**将Web站点直接接入因特网**。



**因特网就是将端系统彼此互联，因此为端系统提供接人的ISP也必须互联。**低层的ISP通过国家的、国际的高层ISP ( 如Level 3 Communications、AT&T、Sprint 和NTT)互联起来。高层ISP是由通过高速光纤链路互联的高速路由器组成的。无论是高层还是低层ISP网络，它们每个都是独立管理的，运行着IP协议(详情见后)，遵从一定的命名和地址习惯。我们将在1.3节中更为详细地考察ISP及其互联的情况。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gewvr9dhodj30rc0ffq7b.jpg)







### **Protocals协议** 

协议控制信息传输规定什么时候送信息，用什么格式送信息，什么时候收信息，收的信息怎么看。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gewxikw7fwj30rb0j9wgh.jpg)

**端系统、 分组交换机**和**其他因特网部件**都要运行-系列协议( protocol),这些协议控制因特网中信息的接收和发送。

TCP ( Transmission Control Protocol,传输控制协议)和IP ( Internet Protocol,网际协议)是因特网中两个最为重要的协议。IP 协议定义了在路由器和端系统之间发送和接收的分组格式。因特网的主要协议统称为TCP/IP。


鉴于因特网协议的重要性，每个人就各个协议及其作用取得一致认识是很重要的， 这样人们就能够创造协同工作的系统和产品。这正是标准发挥作用的地方。






### ***Internet standards***：标准制定组织

•RFC: Request for comments

•IETF: Internet Engineering Task Force

> 国际互联网工程任务组（The Internet Engineering Task Force，简称 IETF）为一个公开性质的大型民间国际团体，汇集了与互联网架构和互联网顺利运作相关的网络设计者、运营者、投资人和研究人员。
>
> RFC，Request For Comments，文件收集了有关互联网相关信息，以及UNIX和互联网社群的软件文件，以编号排定。

因特网标准( Internet standard)由因特网工程任务组( Intermnet Engineering Task Force, IETF)[ IETF
2012]研发。IETF 的标准文档称为请求评论( Request For Comment, RFC)。RFC 最初是
作为普通的请求评论(因此而得名)，以解决因特网先驱者们面临的网络和协议问题[ Allman 2011]。RFC文档往往是技术性很强并相当详细的。它们定义了TCP、IP、HTTP(用于Web)和SMTP (用于电子邮件)等协议。目前已经有将近6000个RFC。其他组织也在制定用于网络组件的标准，最引人注目的是针对网络链路的标准。例如，IEEE 802
LAN/MAN标准化委员会[ IEEE 802 202]制定了以太网和无线WiFi的标准。



## 从 '所提供服务' 的角度总览互联网



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1geww8vka5sj30sq0jhafc.jpg)

前面的讨论已经辨识了构成因特网的许多部件。但是我们也能从一个完全不同的角度，即从为应用程序提供服务的基础设施的角度来描述因特网。

这些应用程序包括电子邮件、Web冲浪、即时讯息、社交网络、IP 语音(VoIP)、流式视频、分布式游戏、对
等(peer-to-peer, P2P) 文件共享、因特网电视、远程注册等等。这些应用程序称**为分布式应用程序( distributed application), 因为它们涉及多台相互交换数据的端系统**。重要的是，因特网应用程序运行在端系统上，即它们并不运行在网络核心中的分组交换机中。尽管分组交换机促进端系统之间的数据交换，但它们并不关心作为数据的源或宿的应用程序。





与因特网相连的**端系统**提供了一个应用程序编程接口( Application Programming Interface,API)，该API规定了运行在一个端系统上的**软件请求因特网基础设施向运行在另一个端系统上的特定目的地软件交付数据的方式**。因特网API是一套发送软件必须遵循的规则集合，因此因特网能够将数据交付给目的地。我们将在第2章详细讨论因特网API。

此时，我们做一一个简单的类比，**在本书中我们将经常使用这个类比**。

假定Alice 使用邮政服务向Bob发一封信。当然，Alice 不能只是写了这封信( 相关数据)然后把该信丢出窗外。相反，邮政服务要求Alice 将信放人一一个信封中;在信封的中央写上Bob的全名、地址和邮政编码;封上信封;在信封的右上角贴上邮票;最后将该信封丢进一个邮局的邮政服务邮箱中。因此，该邮政服务有自己的“邮政服务API"或一套规则， 这是Alice必须遵循的，这样邮政服务才能将自己的信件交付给Bob。

同理，因特网也有一个发送数据的程序必须遵循的API,使因特网向接收数据的程序交付数据。当然，邮政服务向顾客提供了多种服务，如特快专递、挂号、普通服务等。同样的，因特网向应用程序提供了多种服务。当你研发一种因特网应用程序时，也必须为你的应用程序选择其中的一种因特网服务。



