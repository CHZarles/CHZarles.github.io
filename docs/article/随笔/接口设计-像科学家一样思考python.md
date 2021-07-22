---
title: 接口设计-《像科学家一样思考python》
comments: true
copyright: false
date: 2020-01-06 16:57:37
tags: 
	- 读书笔记
	- 《像科学家一样思考python》
categories:
	- 随笔
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



### 以turtle库引入话题

首先了解一下下面代码，自己在电脑写一写，运行一下。

下面是关于turtle库的一些基本用法

```python
import turtle
bob = turtle.Turtle()

print(bob)

#obj.delay 代表每走一步停顿的时间,默认是1s
bob.delay = 2

#turtle移动100个像素(piexl)的距离，方向默认向右
bob.fd(100)
#obj.bk，倒着走200个像素。注意:"朝向没变"。
bob.bk(100)
#obj.lt和obj.rt分别是左转和右转,参数是角度,角度制。
bob.fd(100)
bob.lt(90)
bob.fd(100)

#等待用户进行操作
turtle.mainloop()

```

<!--more-->


运行结果

![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1gamy93uwz5j31h70rhn1h.jpg)



### 封装-泛化-接口设计-重构

```tcl
个人理解，这些东西其实就是：

#----------------------封装（写函数）
#----------------------泛化（增加参数）
#----------------------接口设计（合理设计参数）
#----------------------重构（重新思考代码的复用，重新组织函数关系）
#----------------------文档字符串(docstring)：在函数开头解释接口作用

```



#### 封装

我要画一个正方形，我可以写一个循环，让乌龟画正方形。但下次我要画正方习形时，还要再写一遍。如果，把 实现"画正方形" 这个功能的代码，写成函数，那么下次想画正方形，直接调用函数就行了，这种思想就叫"封装"

```python

#写一个让乌龟画正方形的函数
def square(tmp):
	for i in range(4):
		tmp.fd(100)
		tmp.lt(90)

```



#### 泛化

在上面，我们得到了一个画正方形的函数，但这个函数画的正方向的边长是固定的，并不通用。我们不满足于此，我们想改进这个函数，让这个函数更通用，这种把函数改得更通用的思想，就叫"泛化"。



例如：

1.给square再添加一个形参，控制长度

```python
def square_1(tmp,l):
	for i in range(4):
		tmp.fd(l)
		tmp.lt(90)
```

2.再suqare_1基础上添加形参n，画n边形

```python
def polygon(tmp,n,l):
	for i in range(n):
		tmp.fd(l)
		tmp.lt(360/n)
```



3.改进polygon，画圆，参数是半径R和turtle.Turtle()

```python
#n越大越圆，画的越慢
def circle(tmp,R):
	#R转化为对应的L
	#分割n次,每条边长度为0.5左右
	c = 2*math.pi*R
	n = int(c/0.5)+1
	l = c/n
	polygon(tmp,n,l)
```

4.改进circle,画弧度

```python
def arc(tmp,R,angle):
	#R转化为对应的L
	#分割n次,每条边长度为0.5左右
	c = 2*math.pi*R*(angle/360)
	n = int(c/0.5)+1
	l = c/n
	#画
	for i in range(n):
		tmp.fd(l)
		tmp.lt(angle/n)
```



#### 接口设计

用户通过往函数传入参数，使用我们设计的函数。这些参数，就叫接口。在这个例子里面，设计形参，就叫接口设计。我们写的函数的参数要尽可能少。



#### 重构

**用一句废话来概括就是：重新组织程序，以改善接口，提高代码复用。**



还是举个例子

我们写的那么多函数中，理论上，arc是最基本的，可以演化出画圆。同样，画多边形的函数，可以演化出画正方形的函数。这样写会让接口之间的关系更清晰。

```python
#重构画正方形的函数
def square_2(tmp,l):
     polygon(tmp,4,l)
	

#重构画圆的函数
def circle_1(tmp,R):
    arc(tmp,R,360)   
```

​	



### 编写文档字符串

我们写函数，提供接口让人调用，要让别人知道这些接口的用法。

在python中，文档字符串，实现了这个需求。



看个例子

```python
def arc(tmp,R,angle):
	'''
	R是圆弧对应圆的半径
	angle是圆弧的角度
	'''
	#R转化为对应的L
	#分割n次,每条边长度为0.5左右
	c = 2*math.pi*R*(angle/360)
	n = int(c/0.5)+1
	l = c/n
	#画
	for i in range(n):
		tmp.fd(l)
		tmp.lt(angle/n)
```

函数下面的这个东西就是文档字符串（其实就是注释，但这个注释位置很特殊）

```python
'''
	R是圆弧对应圆的半径
	angle是圆弧的角度
'''
```

查看文档字符串

```python
#打印文档字符串
print(arc.__doc__)
```



### 部分完整代码: 

[basic.py](/download/basic.py)

[practica.py](/download/practica.py)



------

 