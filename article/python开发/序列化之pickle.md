---
title: 序列化之pickle-用python于存档
toc: true
copyright: true
comments: true
date: 2020-08-05 21:28:16
keyword:
tags:
categories: python开发相关
top:
---



## Pickle的方法

pickle的方法的样式基本和json一样

**pickle格式**

```python
import pickle
dic={'name':'alvin','age':23,'sex':'male'} 
print(type(dic)) #<class 'dict'>
```

<!--more-->

**序列化**

```python
j=pickle.dumps(dic) # pickle格式赋值给j 
print(type(j)) #<class 'bytes'> pickle格式就是byte

f=open('序列化对象_pickle','wb')#注意是w是写入str,wb是写入bytes,j是'bytes'
f.write(j)  #-------------------等价于pickle.dump(dic,f)
f.close()
```

**反序列化**

```python
#-------------------------反序列化
import pickle
f=open('序列化对象_pickle','rb')
data=pickle.loads(f.read())#  等价于data=pickle.load(f)
print(data['age'])
```



## 兼容问题

Pickle的问题和所有其他编程语言特有的序列化问题一样，就是它只能用于Python，并且可能不同版本的Python彼此都不兼容，因此，只能用Pickle保存那些不重要的数据，不能成功地反序列化也没关系。 

```python
# coding:utf-8
import pickle

with open('a.pkl',mode='wb') as f:
    # 一：在python3中执行的序列化操作如何兼容python2
    # python2不支持protocol>2，默认python3中protocol=4
    # 所以在python3中dump操作应该指定protocol=2
    pickle.dump('你好啊',f,protocol=2)

with open('a.pkl', mode='rb') as f:
    # 二：python2中反序列化才能正常使用
    res=pickle.load(f)
    print(res)

python2与python3的pickle兼容性问题
```

