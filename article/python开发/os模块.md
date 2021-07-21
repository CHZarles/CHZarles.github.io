---
title: 与操作系统交互的接口-os模块
comments: true
copyright: true
cover: 'https://tva1.sinaimg.cn/large/87c01ec7gy1fsnqqi27uej21kw0w04cl.jpg'
date: 2020-08-04 10:10:13
tags:
categories: python开发相关
photo:
top:
toc: true
---

## 方法总览

```python
# os.getcwd() 获取当前工作目录，即当前python脚本工作的目录路径
# os.chdir("dirname")  改变当前脚本工作目录；相当于shell下cd
# os.curdir  返回当前目录: ('.')
# os.pardir  获取当前目录的父目录字符串名：('..')
# os.makedirs('dirname1/dirname2')    可生成多层递归目录
# os.removedirs('dirname1')    若目录为空，则删除，并递归到上一级目录，如若也为空，则删除，依此类推
# os.mkdir('dirname')    生成单级目录；相当于shell中mkdir dirname
# os.rmdir('dirname')    删除单级空目录，若目录不为空则无法删除，报错；相当于shell中rmdir dirname
# os.listdir('dirname')    列出指定目录下的所有文件和子目录，包括隐藏文件，并以列表方式打印
# os.remove()  删除一个文件
# os.rename("oldname","newname")  重命名文件/目录
# os.stat('path/filename')  获取文件/目录信息
# os.sep    输出操作系统特定的路径分隔符，win下为"\\",Linux下为"/"
# os.linesep    输出当前平台使用的行终止符，win下为"\t\n",Linux下为"\n"
# os.pathsep    输出用于分割文件路径的字符串 win下为;,Linux下为:
# os.name    输出字符串指示当前使用平台。win->'nt'; Linux->'posix'
# os.system("bash command")  运行shell命令，直接显示
# os.environ  获取系统环境变量
# os.path.abspath(path)  返回path规范化的绝对路径
# os.path.split(path)  将path分割成目录和文件名二元组返回
# os.path.dirname(path)  返回path的目录。其实就是os.path.split(path)的第一个元素
# os.path.basename(path)  返回path最后的文件名。如何path以／或\结尾，那么就会返回空值。即os.path.split(path)的第二个元素
# os.path.exists(path)  如果path存在，返回True；如果path不存在，返回False
# os.path.isabs(path)  如果path是绝对路径，返回True
# os.path.isfile(path)  如果path是一个存在的文件，返回True。否则返回False
# os.path.isdir(path)  如果path是一个存在的目录，则返回True。否则返回False
# os.path.join(path1[, path2[, ...]])  将多个路径组合后返回，第一个绝对路径之前的参数将被忽略
# os.path.getatime(path)  返回path所指向的文件或者目录的最后存取时间
# os.path.getmtime(path)  返回path所指向的文件或者目录的最后修改时间
# os.path.getsize(path) 返回path的大小
```

<!--more-->

## 重点方法

```python
#主要是控制操作系统 ，写一些操作系统的控制脚本
import os

res = os.listdir() #return list
# 列出指定目录下的所有文件和子目录，包括隐藏文件，并以列表方式打印
res = os.listdir('.') #列出上一个目录的
print(res)



#删除
 os.remove('re')  #默认当前目录
#改名
 os.rename("input","input.txt") #默认在当前目录找

#获取文件信息 权限 用户组 文件大小
print(os.stat('input.txt'))

#让操作系统的shell运行该命令
os.system(" echo 'Hello World !'")

#获取环境变量
# 环境变量是个统称 ， 该变量是全局性 ，整个软件环境都有效
# 安装pyhon时候 PATH = 文件夹 操作系统找命令时候 在这些变量里找
# sys.path = [] 列表里存的是文件夹 python导入模块时候 找这里面的文件夹
# os.environ 是一个限定的字典，key和val均为字符串
res = os.environ

#在一个文件中  如果你想让某个运行结果 能让整个软件都能用得到
# 你就要让整个结果变成环境变量 加入os.environ
#像字典一样操作 os.environ
os.environ['新添加的KEY'] = 'new'
print(res)


#os.path系列
# os.path.abspath(path)  返回path规范化的绝对路径
print(os.path.abspath(__file__)) #和print(__file__)


#专门获取最后一个斜杠前面的
res = os.path.dirname(r'C:\Users\Charles\AppData\Roaming\Microsoft\Windows\Network Shortcuts')
print(res)
#专门获取路径文件名/或path的最后一个文件夹
res = os.path.basename(r'C:\Users\Charles\AppData\Roaming\Microsoft\Windows\Network Shortcuts')
print(res)


#判断path对应的是一个存在的文件
print(os.path.isfile('a/b/c'))
#判断是不是一个存在的文件夹
print(os.path.isdir(r'C:\Users\Charles'))

#以一个根目录为起始 拼接目录 window的根目录是盘符
print(os.path.join('a','H:\\','b','c')) #输出: 'H:\b\c'

# os.path.getsize(path) 返回path的大小
# 计算文件夹的大小
size = os.path.getsize('.') #返回值是int 单位是字节 传入一个文件夹
print(size)

# 路径处理推荐方法
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```














