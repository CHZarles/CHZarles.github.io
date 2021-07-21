---
title: 数据加密/映射-hashlib模块
toc: true
copyright: false
comments: true
date: 2020-08-06 07:46:23
keyword:
tags:
categories: python开发相关
top:
---



## 什么是哈希hash

**hash是一种算法（python 3.x里代替了md5模块和sha模块，主要提供 SHA1, SHA224, SHA256, SHA384, SHA512 ，MD5 算法），该算法接受传入的内容，经过运算得到一串hash值。**

**HASH值的特点是：**

-  只要传入的内容一样，得到的hash值必然一样 -----》要用明文传输密码文件完整性校验
-  不能由hash值返解成内容- -----》把密码做成hash值，不应该在网络传输明文密码
-  只要使用的hash算法不变，无论校验的内容有多大，得到的hash值长度是固定的

<!--more-->

 hash算法就像一座工厂，工厂接收你送来的原材料（可以用m.update()为工厂运送原材料），经过加工返回的产品就是hash值。

![img](https://gitee.com/chzarles/images/raw/master/imgs/1036857-20180410101832069-1144106861.png)



## hash的用途

**1.hash加密**

```python
#demo
#客户注册,将用户输入的密码以hash字符串形式存到服务器
123465abc ==md5==> hash字符串
#客户登陆,将用户当前输入的密码加密成hash字符串,再验证
客户端 ============hash字符串============》服务端
									  hash字符串
```

**可以用撞库（暴力）的方式破解hash**



**2.hash校验**

用来判断传入的文件是不是一样，用于判断下载的文件是否完整，以及判断下载的文件是否被篡改

```python
客户端<----------------------------服务端
			 接获数据		    
			 篡改数据
```



## hashlib模块

```python
# hashlib

import hashlib
#选择hash算法,hashlib提供了很多算法 这里以md5为例
# m 是 <md5 HASH object @ 0x01CF3E80>
# 就算没有放置要加密的数据 m初始也有对应的hash值
m = hashlib.md5()



# updata(argv) 方法加密, argv必须是byte类型
# 可以多次放入加密内容
m.update('hello'.encode('utf-8'))
m.update('world'.encode('utf-8'))

#hexdigest()获取加密内容
res = m.hexdigest()
print(res)
```

> 注意：把一段很长的数据update多次，与一次update这段长数据，得到的结果一样
> upadate多次会让效率变低,但是update多次为校验大文件提供了可能。
>
> 当然,在校验大文件的同时还想保存效率,可以采取一下策略
> 1.利用seek() 随机取文件的几段数据流 update相关数据 (要保存seek移动的节点轨迹)
> 2.客户段 根据seek移动的节点轨迹 进行校验





## 模拟撞库

```python
# 模拟撞库
import hashlib
#抓获到的密文
cryptograph='aee949757a2e698417463d47acac93df'
#可能的密码
passwds=[
    'alex3714',
    'alex1313',
    'alex94139413',
    'alex123456',
    '123456alex',
    'a123lex',
    ]

#制作密码字典
def make_passwd_dic(passwds):
    dic={}
    for passwd in passwds:
        m=hashlib.md5()
        m.update(passwd.encode('utf-8'))
        dic[passwd]=m.hexdigest()
    return dic

#暴力对比
def break_code(cryptograph,passwd_dic):
    for k,v in passwd_dic.items():
        if v == cryptograph:
            print('密码是===>\033[46m%s\033[0m' %k)


#撞库函数
break_code(cryptograph,make_passwd_dic(passwds))
```

> 应对撞库.提升撞库的成本-->密码加盐:就是往原来的数据里掺入无用的干扰数据来加密