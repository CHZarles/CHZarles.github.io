---
title: 执行系统命令-subprocess模块
toc: true
copyright: true
comments: true
date: 2020-08-06 09:06:06
keyword:
tags:
categories: python开发相关
top:
---



## subprocess模块

```python
import subprocess
#用来执行系统命令
#开启一个子进程来运行相关程序

#打开shell
#正确运行的命令结果存在stdout定向的管道
#错误运行的命令结果存在stderr定向的管道
obj = subprocess.Popen(r'dir c:\User\Charles',shell=True,
                 stdout=subprocess.PIPE,
                 stderr=subprocess.PIPE,
                 )

res = obj.stdout.read() #res是byte格式
print(res.decode('GBK')) #要用windows系统的编码来解码

输出:

res = obj.stderr.read() #res是byte格式
print(res.decode('GBK')) #要用windows系统的编码来解码

输出:系统找不到指定的文件。
```

<!--more-->