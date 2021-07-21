---
title: 与python解释器交互的一个接口-sys模块
comments: true
copyright: false
cover: 'https://tva1.sinaimg.cn/large/87c01ec7gy1fsnqqi27uej21kw0w04cl.jpg'
date: 2020-08-04 10:16:48
tags:
categories: python开发相关
photo:
top:
toc: true
---



## sys.argv方法

```python
# sys模块
import sys
# 接受命令参数
# sys.argv获取的是解释器后的参数值 返回一个列表
# 命令行参数List，第一个元素是程序本身路径
l = sys.argv
print(l)

if len(l) > 1 and l[1] == 'A':
    print('获取参数A')
else:
    print("参数错误")
    

输出:
['C:/Users/Charles/PycharmProjects/练习/sys模块.py']
参数错误
```

<!--more-->



## Demo

```python
#文件复制小程序demo
import sys
import os
l = sys.argv
# 输入合法性判断
# 1.判断有没有输入两个参数
if len(l) != 3:
    print("参数错误")
    sys.exit()
# 2.判断路径合法性
if not os.path.isfile(l[1]):
    print("源文件不存在")
    sys.exit()


src_file = sys.argv[1]
dst_file = sys.argv[2]
with open(r'%s' %src_file,mode = 'rb') as read_file,\
    open(r'%s' %dst_file,mode = 'wb') as write_file:
    for line in read_file:
        write_file.write(line)
```

测试

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ghekf6izq0j30qu05igm1.jpg)



## sys.exit(n)方法

功能：执行到主程序末尾，解释器自动退出，但是如果需要中途退出程序，可以调用sys.exit函数，带有一个可选的整数参数返回给调用它的程序，表示你可以在主程序中捕获对sys.exit的调用。（0是正常退出，其他为异常）



## sys.path

功能：获取指定模块搜索路径的字符串集合，可以将写好的模块放在得到的某个路径下，就可以在程序中import时正确找到。

示例：

```python
>>> import sys
>>> sys.path
['', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/python2.7/dist-packages/ubuntu-sso-client']
sys.path.append("自定义模块路径")
```

## sys.modules

功能：`sys.modules`是一个全局字典，该字典是python启动后就加载在内存中。每当程序员导入新的模块，`sys.modules`将自动记录该模块。当第二次再导入该模块时，python会直接到字典中查找，从而加快了程序运行的速度。它拥有字典所拥有的一切方法。

示例：`modules.py`

```python
#!/usr/bin/env python
import sys
print sys.modules.keys()
print sys.modules.values()
print sys.modules["os"]
```

运行：

```python
python modules.py
['copy_reg', 'sre_compile', '_sre', 'encodings', 'site', '__builtin__',......
```

## sys.stdin\stdout\stderr

功能：stdin , stdout , 以及stderr 变量包含与标准I/O 流对应的流对象. 如果需要更好地控制输出,而print 不能满足你的要求, 它们就是你所需要的. 你也可以替换它们, 这时候你就可以重定向输出和输入到其它设备( device ), 或者以非标准的方式处理它们




