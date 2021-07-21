---
title: Bresenham画直线and中点画圆法
comments: true
copyright: false
toc: true
date: 2020-09-24 23:17:32
tags:
categories: 计算机视觉
photo:
top:
cover:
---





## 先导知识

[Python--Matplotlib（基本用法）](https://blog.csdn.net/qq_34859482/article/details/80617391)

[Bresenham直线算法与画圆算法](https://oldj.net/blog/2010/08/27/bresenham-algorithm)



## 代码

```python
from matplotlib import pyplot as plt

def drawline(points):
    x1,y1,x2,y2= map(int,points)
    px=[]#用来保存直线各个点的x坐标
    py=[]#用来保存直线各个点的y坐标
    p = 2*(y2-y1)-(x2-x1) #
    delx = x2-x1
    dely = y2-y1

    y = y1
    for x in range(x1,x2+1):
        px.append(x)
        py.append(y)
        if p>0:
            y+=1
            p = p+2*(dely-delx)
        else:
            p = p+2*dely

    plt.scatter(px, py, color='red', marker='.')#根据点输出直线
    # plt.show()

def drawcirl(r):
    x=0
    y=r
    e=1-r
    # 对称的八个点全存进去
    px.extend([x, y, -x, y, x, -y, -x, -y])
    py.extend([y, x, y, -x, -y, x, -y, -x])
    while x<=y:
        if e<0:
            e+=2*x+3
        else:
            e+=2*(x-y)+5
            y-=1
        x+=1
        # 对称的八个点全存进去
        px.extend([x,y,-x,y,x,-y,-x,-y])
        py.extend([y,x,y,-x,-y,x,-y,-x])
    plt.scatter(px, py, color='blue', marker='.')


# x1,y1,x2,y2= map(int,input('输入x1,y1,x2,y2空格隔开(x1<x2):').split())
io =[-1,-5,10,10]
drawcirl(5)
drawline(io)

plt.show()
```

<!--more-->

## 效果

![image-20201010234221301](https://gitee.com/chzarles/images/raw/master/imgs/image-20201010234221301.png)



