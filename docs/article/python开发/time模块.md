---
title: time模块
comments: true
copyright: true
cover: 'https://tva1.sinaimg.cn/large/87c01ec7gy1fsnqqi27uej21kw0w04cl.jpg'
date: 2020-08-04 09:52:33
tags:
categories: python开发相关
photo:
top:
toc: true
---



## Time模块

**时间分为三种格式**

**1.时间戳**

```python
# 1.时间戳:从1970年到现在经过的秒数
# 作用：时间戳主要用于时间间隔的计算
print(time.time())

输出：
1596363355.6840038
```

<!--more-->

**2.结构化时间**

```python
#作用：获取当前时间的某一部分
res = time.localtime()
print(res)
#time.struct_time(tm_year=2020, tm_mon=8, tm_mday=2, tm_hour=18, tm_min=15, tm_sec=55, tm_wday=6, tm_yday=215, tm_isdst=0)
#tm_wday=6 一个星期的第六天, tm_yday=215 今年的第215填, tm_isdst=0

#获取当前年份
print(res.tm_year) #2020
#获取今天是今年的第几天
print(res.tm_yday)#215
```



**3.字符串格式化时间**

```python
# 作用：主要用于展示时间
print(time.strftime('%Y-%m-%d %H:%M:%S %p'))
print(time.strftime('%Y-%m-%d %X')) #大小写有别
# 2020-08-02 18:15:55 PM
# 2020-08-02 18:15:55
```



## Datetime模块

```python
#这个模块的时间可以参与运算
import datetime
print(datetime.datetime.now())
print(datetime.datetime.now()+datetime.timedelta(days=3)) #加三天
print(datetime.datetime.now()+datetime.timedelta(weeks=-1)) #减一个星期

#相关参数
# days: float
# seconds: float
# microseconds: float
# hours: float
# weeks: float
# fold: int
```





## 时间格式之间的转化

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1ghejundojaj31090lsdlm.jpg)

```python
#1. 时间格式的转换
#struct_time->时间戳
import time
s_time = time.localtime()
print(time.mktime(s_time))

#时间戳->struct_time
# way1
tp_time = time.time()
print(time.localtime(tp_time))


#struct_time->格式化字符串时间
s_time = time.localtime()
print(time.strftime('%Y-%m-%d %H:%M:%S ',s_time))

#格式化字符串时间 ->struct_time
print(time.strptime('2020-08-02 19:36:05','%Y-%m-%d %H:%M:%S'))
```



## 常见应用

```python
#demo文本有此时间 在此时间基础上加7天
'1988-03-03 11:11:11'
```

做法

```python
#先转结构化时间
struct_time = time.strptime('1988-03-03 11:11:11','%Y-%m-%d %H:%M:%S')
#再转时间戳 顺便加7天
timestamp = time.mktime(struct_time) + 7*3600*24


#还是先转结构化时间
struct_time = time.localtime(timestamp)
#结构化转格式化字符串
format_time = time.strftime('%Y-%m-%d %H:%M:%S ',struct_time)
print(format_time)
```