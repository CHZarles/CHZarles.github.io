---
title: 网络编程-实现-远程执行命令的CS架构软件
comments: true
copyright: false
toc: true
date: 2020-08-29 14:28:11
tags:
categories: python开发相关
photo:
top:
cover:
---

## 服务端

**这个软件用到的是TCP协议，TCP协议在编程中会有粘包问题**

如果用UDP协议就没有粘包问题，但是，UDP协议不能发很大的数据，而且也没有TCP数据那么可靠

```python
# 服务端应该满足的特点：
# 1. 一直提供服务
# 2.并发提供服务

import subprocess
import struct
from socket import *
# #1.实例化socket对象: socket.AF_INET族,socket.SOCK_STREAM流式协议 => TCP协议
server = socket(AF_INET,SOCK_STREAM)

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
            data = conn.recv(8096)  #最大的接受数据量为8096byte   写多大也没意义
            if len(data) == 0:
                #tcp协议不能发空 不能收空
                #在unix系统 一旦data收到的是空
                #意味着一种异常行为：客户非法断开链接
                #写这个为了防止客户端突然中断
                break
            print('客户端发来:',data.decode('GBK'))
            obj = subprocess.Popen(data.decode('GBK'),
                             shell = True,
                             stdout = subprocess.PIPE,
                             stderr = subprocess.PIPE,
                             )
            stdout_res = obj.stdout.read()  # 返回byte格式
            stderr_res = obj.stderr.read()  # 返回是byte格式

            total_size=len(stderr_res)+len(stdout_res)
            #1.先发头信息（用hash的思想，把整数转化为固定长度的byte): 对数据描述信息
            # 这里要用到struct模块 int-》固定长度的byte
            header = struct.pack('i',total_size)  #参数为i时 ，发送4个byte
            conn.send(header)

            #2.再发真实数据
            #send是把数据发到网卡 后续工作由操作系统完成
            conn.send( stdout_res + stderr_res)
        except Exception:
            #针对windows系统
            break

    #6.关闭连接conn
    conn.close()



# 关闭服务器操作
# phone.close()
```

<!--more-->

## 客户端

```python
import struct
#因为这个程序比较简单 直接用from ... import *
from socket import *
#1.实例化socket对象: socket.AF_INET族,socket.SOCK_STREAM流式协议 => TCP协议
cli = socket(AF_INET,SOCK_STREAM)

#2.给服务器发连接请求
# 写服务端的IP+端口
cli.connect(('127.0.0.1',8080))

#3.通信:收/发消息
while True:
    msg = input("输入指令>>>: ").strip() #msg = ''
    # 要解决发送空的问题
    if len(msg) == 0:
        continue
    cli.send(msg.encode('GBK'))


    # 解决粘包问题思路:
    # 1、拿到数据的总大小total_size
    header = cli.recv(4)
    total_size = struct.unpack('i', header)[0]
    # 2、recv_size=0，循环接收，每接收一次，recv_size+=接收的长度
    # 3、直到recv_s1ze=total_size

    recv_size = 0
    print('服务器发来运行结果:')
    while recv_size < total_size:
        recv_data = cli.recv(1024)  # 接受结果 本次接收最大接收1024个字节 ’流‘会出现粘包问题
        recv_size+=len(recv_data)
        print(recv_data.decode('GBK'))  # window系统解码成GBK




# 关闭连接
cli.close()
```



[关于struct模块和其它网络编程示例](https://www.cnblogs.com/linhaifeng/articles/6129246.html#_label12)






