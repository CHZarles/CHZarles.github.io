---
title: python中递归的坑
comments: true
copyright: false
cover: 'https://tva1.sinaimg.cn/large/87c01ec7gy1fsnqqi27uej21kw0w04cl.jpg'
date: 2020-08-01 21:09:06
tags:
categories: python基本语法
photo:
top:
toc: true
---



## 导入

看一个代码运行案例

```python
def check(x):
    if x>=1024:
        return x
    x*=2

print(check(2))

输出:
None
```

返回值居然是None?

离谱 ，按照我以往学习c++的经验 ，返回值感觉应该是一个数(此处调用返回的结果应该是1024)

![image.png](https://i.loli.net/2021/07/22/oJZxr132KSzuYhQ.jpg)

python没有伪递归优化，不应该让递归无限大调用。

幸好找到了解答

[Python 递归函数返回值为None的解决办法](https://blog.csdn.net/ha_hha/article/details/79393041)

<!--more-->













