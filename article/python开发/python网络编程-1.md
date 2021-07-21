---
title: python网络编程入门
comments: true
copyright: false
toc: 
date: 2020-08-25 14:48:42
tags:
categories: python开发相关
photo:
top:
cover:
---

## 计算机网络关键词回顾

```python
# 1 CS架构与BS架构
Client  <==========>    Server
客户端软件send           服务端软件recv
操作系统                 操作系统
计算机硬件<=== =物理介质====>计算机硬件

Browser<==========>Server

#1物理层负责发送电信号
一组物理层数据称之为:位
单纯的电信号毫无意义，必须对其进行分组

#2数据链路层: ethernet以太网协议
规定1:一组数据称之为一个数据帧
规定2:数据帧分成两部分=》头+数据
     头包含:源地址与目标地址，该地址是mac地址
     数据部分:网络层发来的内容
规定3:规定但凡接入互联网的主机必须有一块网卡，每块网卡在出厂时都烧制好一个全世界唯一的mac地址

注意:计算机通信基本靠吼，即以太网协议的工作方式是广播（是以太网，以太网！！！同一个广播域内！同一个网络内）

#3网络层: IP协议
要达到的目的:划分广播域
每一个广播域但凡要接通外部，一定要有一个网关帮内部的计算机转发包到公网
网关与外界通信走的是路由协议
    规定1:一组数据称之为一个数据包
    规定2:数据帧分成两部分=》头+数据
    头包含:源地址与目标地址,该地址是IP地址
    数据包含的:传输层发来的数据

IP地址必须和子网掩码配合使用 =》 用来找网关 =》找对应局域网


计算机1:                           计算机2:
应用层                             应用层
传输层                             传输层
网络层                             网络层
数据链路层                        数据链路层
物理层 《=========二/三层交互机========》 物理层
                                0101010101010

计算机1的数据刚传到数据链路层时候的状态:
(源mac地址(计算机1的)，XXXX(计算机2的))+(源ip地址(计算机1的)， 目标ip地址(计算机2的))+数据



交换机只事先知道的是对方的ip地址
但是计算机的底层通信是基于ethernet以太网协议的mac地址通信
而且交换机并不知道要往哪个目标转发数据(这个目标可能是另一个交换机或者网关....)
要借助ARP协议

ARP: 能够将ip地址解析成mac地址
ARP工作原理:https://www.cnblogs.com/linhaifeng/articles/5937962.html
同局域网 直接发
不同局域网 发给网关

总结：
（公网）IP地址用来定义到子网
一般mac只在局域网起作用，mac用来标识局域网内的唯一一个机器
又因为arp协议
ip地址 => 标识全世界范围独一无二的计算机

#4 传输层： TCP/UDP协议
所有涉及网络通信的软件都有端口
ip + mac + 端口 可以标识全世界所有软件
端口范围0-65535，0-1023为系统占用端口
基于tcp协议通信之前:必须建立一个双向通信的链接
C-—---—-------------->S
C<------——-----——-----S

三次握手建立链接:
建立链接是为了传数据做准备的，三次握手即可
四次挥手断开链接
断开链接时，由于链接内有数据传输，所以必须分四次断开

tcp传输是可靠的


# 5.应用层:
可以自定义协议=》头部+数据部分
自定义协议需要注意的问题:
1、两大组成部分=头部+数据部分
头部:放对数据的描述信息
比如:数据要发给谁，数据的类型，数据的长度
数据部分:想要发的数据
2、头部的长度必须固定
因为接收端要通过头部获取所接接收数据的详细信息

常用的熟知的协议 http https ftp

socket:对传输层一下的操作进行了封装
我们写程序只要根据协议构造应用层的数据就行了
```

<!--more-->



## TCP协议服务器/客户端-非多进程写法

`服务器.py`

```python
# 服务端应该满足的特点：
# 1. 一直提供服务
# 2. 并发提供服务

import socket
# #1.实例化socket对象: socket.AF_INET族,socket.SOCK_STREAM流式协议 => TCP协议
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#2.绑定服务器地址: 绑定自己的IP地址，端口
# 用IPconfig看到的只是私网IP
# 绑定私网IP 只能让局域网内的服务器访问 没什么意义
# 推荐测试的时候绑定 127.0.0.1 只能让自己的机器绑定
# 部署到服务器时候，绑定 0.0.0.0
# 1024以前的端口被系统占用
server.bind(('127.0.0.1',8080))

# 3.等待/储存 链接请求
server.listen(5) #5指的是半连接池大小

# 4.接受请求
# 链接循环
while True:
    # 返回值是连接对象,连接对象地址
    # 从半连接池获取一个连接请求
    conn,client_addr = server.accept()
    print('接受一个请求')
    print(conn,':',client_addr)

    #5.通信:收/发消息
    while True:
        try:
            # recv 是从网卡的缓存拿数据
            data = conn.recv(1024)  #最大的接受数据量为1024byte
            if len(data) == 0:
                #tcp协议不能发空 不能收空 这和tcp是流式协议的特性有关
                #在unix系统 一旦data收到的是空
                #意味着一种异常行为：客户非法断开链接
                #写这个为了防止客户端突然中断
                break
            print('客户端发来:',data.decode('utf-8'))
            #send是把数据发到网卡 后续工作由操作系统完成
            conn.send(data.upper())
        except Exception:
            #针对windows系统
            break

    #6.关闭连接conn
    conn.close()



# 关闭服务器操作
# phone.close()
```

## 

`客户端.py`

```python
import socket
#1.实例化socket对象: socket.AF_INET族,socket.SOCK_STREAM流式协议 => TCP协议
cli = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#2.给服务器发连接请求
# 写服务端的IP+端口
cli.connect(('127.0.0.1',8080))

#3.通信:收/发消息
while True:
    msg = input("输入要发送的消息>>>: ").strip() #msg = ''
    # 要解决发送空的问题
    if len(msg) == 0:
        continue
    cli.send(msg.encode('utf-8'))
    data = cli.recv(1024)
    print('服务器发来:',data.decode('utf-8'))


# 关闭连接
cli.close()
```



## UDP协议服务器/客户端-非多进程写法

`服务器.py`

```python
#UDP协议二者其一中断,都不会影响彼此

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  #数据报协议 =》UDP协议
# 绑定端口:
s.bind(('127.0.0.1', 9999))
#创建Socket时，SOCK_DGRAM指定了这个Socket的类型是UDP。绑定端口和TCP一样，但是不需要调用listen()方法，而是直接接收来自任何客户端的数据：
print('Bind UDP on 9999...')
while True:
    # 接收数据:
    data, addr = s.recvfrom(1024)
    print('Received from %s:%s.' % addr)
    s.sendto(b'Hello, %s!' % data, addr)
```

`客户端.py`

```python
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #数据报协议 =》UDP协议
while True:
    # udp协议可以发空
    data = input('>>>>: ').strip()
    # 发送数据:
    s.sendto(data.encode('utf-8'), ('127.0.0.1', 9999))
    # 接收数据:
    print(s.recv(1024).decode('utf-8'))
s.close()
```