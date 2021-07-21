---
title: python 列表基本操作
comments: true
copyright: false
date: 2020-01-14 23:26:20
tags:
	- python
categories:
	- 随笔
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



## 知识点

#### #列表是可变的,且列表元素不一定要同类型

```python
print('-'*20,"列表可变",'-'*20)
test = ["sss",4]
print(test[0])
print(test[1])
print(test[-2])
```

<!--more-->
#### #列表嵌套

```python
print('-'*20,"列表嵌套",'-'*20)
#列表是序列，且可以嵌套
test[0] = [4,5,6]
print(test[0])
```

#### #in 操作符用于列表

```python
print('-'*20,"in操作符",'-'*20)
print (4 in test)
```

#### #遍历列表

```python
print('-'*20,"遍历列表",'-'*20)
for item in test:
	print(item)
```

#### #列表拼接

```python
print('-'*20,"列表拼接",'-'*20)
tail = ["这是拼接的部分","用+就可实现"]
test = test + tail
print(test)
```

#### #列表重复

```python
print('-'*20,"列表重复",'-'*20)
tail = ["用*实现列表重复"]
tail *= 3
print(tail)
```

#### #列表切片

```python
print('-'*20,"列表切片",'-'*20)
test = test[0:2]
print(test)
```

#### #列表拷贝

reference：[python深拷贝和浅拷贝的区别](https://www.cnblogs.com/xueli/p/4952063.html)

```python
print('-'*20,"浅copy列表",'-'*20)
temp = test.copy()
test[0] = "改了"
print(temp)
print(test)

结果
-------------------- 浅copy列表 --------------------
[1, 2, 3, 4, [1, 2, 3]]
['改了', 2, 3, 4, [1, 2, 3]]

```

改变子对象

```python
print('-'*20,"浅copy列表，子对象共用",'-'*20)
test = [1,2,[1,2,3]]
temp = test.copy()
# 这里就改变了子对象
test[2].append("ss")
print(temp)
print(test)

结果
-------------------- 浅copy列表，子对象共用 --------------------
[1, 2, [1, 2, 3, 'ss']]
[1, 2, [1, 2, 3, 'ss']]


import copy
print('-'*20,"深copy列表，子对象共用",'-'*20)
test = [1,2,[1,2,3]]
temp = copy.deepcopy(test)
# 这里就改变了子对象
test[2].append("ss")
print(temp)
print(test)

结果
-------------------- 浅copy列表，子对象共用 --------------------
[1, 2, [1, 2, 3]]
[1, 2, [1, 2, 3, 'ss']]

```



#### #更新多个列表元素

```python
print('-'*20,"更新多个列表元素",'-'*20)
test[0:2] = ["X","Y"]
print(test)
```

#### #列表方法append

```python
print('-'*20,"append接受元素，在列表尾部添加为元素",'-'*20)
temp = test.copy()
temp.append(["ssr",6,5,4])
print(test)
print(temp)
```

#### #列表方法extend

```python
print('-'*20,"extend 接受列表，合并列表",'-'*20)
temp = test.copy()
temp.extend(["ssr",6,5,4])
print(test)
print(temp)
```

#### #删除元素

```python
print('-'*20,"pop(i) 删除下标为i的元素",'-'*20)
temp = test.copy()
temp.pop(0)
print(test)
print(temp)
```

#### #删除元素

```python
print('-'*20,"remove(op) 删除内容为op的第一个元素",'-'*20)
test = ['x','y','x','y']
temp = test.copy()
temp.remove('x')
print(test)
print(temp)
```

#### #删除多个元素

```python
print('-'*20,"del 切片删除多个元素 ",'-'*20)
test = ['1','2','3','4']
temp = test.copy()
del temp[0:2]
print(test)
print(temp)
```



#### #字符串转列表

```python
print('-'*20,"字符串转列表 ",'-'*20)
test = "string"
temp = list(test)
print(test)
print(temp)
```

#### #字符串转列表

```python
print('-'*20,"字符串 去除分隔符 转列表 ",'-'*20)
test = "this is my string"
temp = test.split()
print(test)
print(temp)

test = "this*is*my*string"
temp = test.split("*")
print(test)
print(temp)
```

#### #列表转字符串

```python
print('-'*20," 添加分隔符 列表 转字符串 ",'-'*20)
test = ["this","is","a","list"]
temp = " ".join(test)
print(test)
print(temp)
```

#### #列表排序

```python
print('-'*20," sort 列表排序,改变原列表 ",'-'*20)
test = ["this","is","a","list"]
temp = test.copy()
temp.sort()
print(test)
print(temp)
```

#### #列表排序

```python
print('-'*20," sorted 列表排序,不改变原列表,返回一个新列表 ",'-'*20)
test = ["this","is","a","list"]
temp = sorted(test)
print(test)
print(temp)
```


