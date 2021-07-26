---
title: CHAPTER2应用层-DNS
comments: true
copyright: false
date: 2020-06-19 23:25:41
tags:
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gfzqokt47cj30ip0ebtbp.jpg
toc: true
---

## DNS概述-域名系统

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfzqokt47cj30ip0ebtbp.jpg)

DNS是互联网的核心服务。解决的是互联网上主机/路由器的识别问题。

互联网上每台主机都有一个地址，即IP地址。IP地址是一串数字，不方便记忆。我们通常用域名来标识主机。

DNS解决的就是IP和域名的映射问题。

**DNS通常是由其他应用层协议所使用的，包括HTTP、SMTP和FTP,将用户提供的主**
**机名解析为IP地址。**

**DNS有以下特点**

- 是多层域名服务器构成的[分布式数据库](https://baike.baidu.com/item/%E5%88%86%E5%B8%83%E5%BC%8F%E6%95%B0%E6%8D%AE%E5%BA%93/1238109?fr=aladdin)
- DNS本身也是应用层协议，能完成名字的解析

**DNS的服务**

- 域名向IP地址翻译(最基本的功能)
- 主机别名:一个主机除了域名外可能还有别名。应用程序可以调用DNS来获得主机别名对应的规范主机名以及主机
  的IP地址。
- 邮件服务器别名：邮件应用程序可以调用DNS解析邮件服务器别名
- 负载均衡:紧忙的站点( 如cnn. com)冗余分布在多台服务器上，每台服务器均运行在不同的端系统上，每个都有着不同的IP地址。由于这些冗余的Web服务器，一个IP地址集合因此与同一个规范主机名相联系。DNS数据库中存储着这些IP地址集合。当客户对映射到某地址集合的名字发出一个DNS请求时，该服务器用IP地址的整个集合进行响应，但在每个回答中循环这些地址次序。因为客户通常总是向IP地址排在最前面的服务器发送HTTP请求报文，所以DNS就在所有这些冗余的Web服务器之间循环分配了负载。





**DNS分布式数据库示意图**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfy1ebezxnj30sa0bfdne.jpg)

<!--more-->

大致来说有三种类型的DNS服务器，根域名服务器，顶级域(TLD)DNS服务器,权威服务器。

> 如果DNS客户要决定主机名www. amazon. com的IP地址。粗略说来，将发生下列事件。客户首先与根服务器之一联系，它将返回顶级域名com的TLD服务器的IP地址。该客户则与这些TLD服务器之一联系， 它将为amazon. com返回权威服务器的IP地址。最后，该客户与amazon. com权威服务器之一联系，它为主机名www. amazon. com返回其IP地址。
>
> 

- 根DNS服务器: 在因特网上有13个根DNS服务器(标号为A到M)，它们中的大部分位于北美洲。尽管我们将这13个根DNS服务器中的每个都视为单个的服务器，但每台“服务器”实际上是一个冗余服务器的网络，以提供安全性和可靠性。

- 顶级域(DNS)服务器: 这些服务器负责顶级域名如com、org、 net、 edu 和gov,以及所有国家的顶级域名如uk、fr、 ca 和jp。Verisign Global Registry Services 公司维护com顶级域的TLD服务器; Educause公司维护edu顶级域的TLD服务器。

- 权威DNS服务器: 互联网上可被公开访问的服务器(如Web服务器和邮件服务器)的每个组织机构必须提供公共可访问的DNS记录，这些记录将这些主机的名字映射为IP地址。一个组织机构的权威DNS服务器收藏了这些DNS记录。

  > 我们可以通过购买服务，让这些记录存储在某个服务提供商的一个权威DNS服务器中。
  >
  > 多数大学和大公司实现和维护它们自己基本和辅助( 备份)的权威DNS服务器。

- 有一类重要的DNS,称为本地DNS服务器(local DNS server): 一个本地DNS服务器严格说来并不属于该服务器的层次结构，但它对DNS层次结构是重要的。每个ISP (如一个大学、一个系、一个公司或一个居民区的ISP)都有一台本地DNS服务器( 也叫默认名字服务器)。当主机与某个ISP连接时，该ISP提供一台主机的IP地址，该主机具有一台或多台本地DNS服务器IP地址机在同一个局域网中

> 对于某居民区ISP来说，本地DNS服务器通常与主机相隔不超过几台路由器。当主机发出DNS请求时，该请求被发往本地DNS服务器，它起着代理的作用，并将该请求转发到DNS服务器层次结构中



## DNS查询方式示例

**迭代查询方式**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfymmieyfdj30px0c1gtl.jpg)



**递归查询方法**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfymp9zw6aj30pt0c1ag6.jpg)



## **DNS记录的缓存和更新问题**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfymswnppzj30pv095n2w.jpg)



## DNS记录与消息格式

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfymvzkzanj30ql0bojx6.jpg)



## DNS协议报文

DNS**是查询回复协议**。这些报文都是可读的。

[DNS的传输层协议](https://www.cnblogs.com/wuyepeng/p/9835839.html)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfymyht7ejj30qf0bjgs2.jpg)

我在wireshark抓取的报文

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfyn89zm2cj30sm081aab.jpg)