---
title: Socketserver模块并发运行程序
comments: true
copyright: false
toc: true
date: 2020-10-07 17:21:43
tags: python
categories: python开发相关
photo:
top:
cover:
---

## 针对TCP协议下的用法

服务端

```python
# 服务端应该满足的特点：
# 1. 一直提供服务
# 2. 并发提供服务

import socketserver

# 通信循环写在这里面
class MyRequestHandle(socketserver.BaseRequestHandler):  #这里是继承
    def handle(self):
        print(self.request)  #self.request 是每个接收的tcp客户端套接字对象
        print(self.client_address);

        # 通信循环:收/发消息
        while True:
            try:
                # recv 是从网卡的缓存拿数据
                data = self.request.recv(1024)  # 最大的接受数据量为1024byte
                if len(data) == 0:
                    # tcp协议不能发空 不能收空 这和tcp是流式协议的特性有关
                    # 在unix系统 一旦data收到的是空
                    # 意味着一种异常行为：客户非法断开链接
                    # 写这个为了防止客户端突然中断
                    break
                print('客户端发来:', data.decode('utf-8'))
                # send是把数据发到网卡 后续工作由操作系统完成
                self.request.send(data.upper())
            except Exception:
                # 针对windows系统
                break

        # 关闭连接conn
        self.request.close()

s = socketserver.ThreadingTCPServer(('127.0.0.1',8880),MyRequestHandle)
s.serve_forever()
```



<!--more-->

客户端

```
import socket
#1.实例化socket对象: socket.AF_INET族,socket.SOCK_STREAM流式协议 => TCP协议
cli = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

#2.给服务器发连接请求
# 写服务端的IP+端口
cli.connect(('127.0.0.1',8880))

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



## 针对UDP协议的用法

服务端

UDP不是面向连接的，看起来像是并发，但是其实不是并发的。

```python
#UDP协议二者其一中断都不会影响彼此

import socketserver

class MyRequestHandle(socketserver.BaseRequestHandler):  #这里是继承
    def handle(self):
        # print(self.request)  #self.request 是一个元组
        client_data = self.request[0]   #获取发来的数据
        server = self.request[1]        #服务端的套接字对象？
        client_addr = self.client_address   #发来数据的地址
        print('客户发来数据%s' %client_data)
        server.sendto(client_data.upper(),client_addr)


s = socketserver.ThreadingUDPServer(('127.0.0.1',9999),MyRequestHandle)
s.serve_forever()
```

客户端

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

