---
title: random模块
comments: true
copyright: false
cover: 'https://tva1.sinaimg.cn/large/87c01ec7gy1fsnqqi27uej21kw0w04cl.jpg'
date: 2020-08-04 10:06:58
tags:
categories: python开发相关
photo:
top:
toc: true
---

## 常用方法

```python
#random模块
import random
print(random.random())  #取0-1之间的浮点数
print(random.randint(1,3)) #[1,3] 取出整数
print(random.randrange(1,3)) #[1,3) 取出整数
print(random.uniform(1,3))#取出(1,3)的浮点数
print(random.choice([1,2,999,'error',['charles','Mary']])) #在自己定制的选项里选择
print(random.sample([1,2,3,4,5,6],3)) #choice的升级版，可取多个样本

item = ['♠A','♦1','♥2','♧Q']
random.shuffle(item) #随机打乱列表里元素的顺序，返回值是None
print(item)
```

<!--more-->



## 应用:生成六位验证码

```python
# 应用：随机验证码
#规定是6位验证码，有字母有数字
# 思路 ：随机挑选六个字符串加起来
import random
code = ''
for i in range(6):
    num = str(random.randint(0,9)) #取0-9的整数 再转string
    ch = chr(random.randint(ord('A'),ord('Z'))) #ord(char) char是长度为1的字符串 ，字母转ascll码
    code += random.choice([num,ch])
print(code)
```










