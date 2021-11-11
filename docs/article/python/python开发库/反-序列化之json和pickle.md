---
title: 序列化之json-用于跨平台交互
toc: true
copyright: true
comments: true
date: 2020-08-05 20:46:26
keyword:
tags:
categories: python开发相关
top:
---

## 什么是序列化和反序列化

- 序列化:将内存的数据格式转换成一个特定的格式 , 该格式的内容可用于存储或者传输给其它平台
- 反序列化:将特定格式反解成其平台对应的格式
- 内存中的数据类型<--->(反)序列化<--->特定的格式(json格式或者pickle格式)

```python
'''
序列化的例子
{'temp':1024} --> 序列化:str({'temp':1024}) --> "{'temp':1024} "
# 反序列化
{'temp':1024} --> 反序列化:eval({'temp':1024}) <-- "{'temp':1024} "
'''
```

<!--more-->

## 为什么要序列化

```
序列化得到的格式有两种用途
  1 用于存储 -》可以保存程序的运行状态(保存变量值)-》用于存档
  2 传输给其它平台使用 -》跨平台数据交互(可能会跨语言)
    python                    Java
     列表  <--> (反)序列化 <--> 数组

  强调:
    针对用途一: 可以是专用的格式，只要程序能识别就行 -》 pickle 只要 python 能识别
    针对用途二: 所有语言都能识别这个格式 -》 提取所有编程语言共有的数据类型 -》 json格式
```



## 如何序列化和反序列化

用到的方法有

- `json.dumps() 和 json.dump()`
- `json.loads() 和 json.load()`

**序列化**

```python
import json
# step1 序列化
json_res = json.dumps([1,'abc'])
print(json_res,type(json_res)) #[1, "abc"]  <class 'str'>
#step2 将序列化结果存到硬盘
with open('test.json',mode='wt',encoding='utf-8') as f:
    f.write(json_res)

#将序列化结果写入文件的简单方法,就是将step1 融入到 step2
with open('test.json',mode='wt',encoding='utf-8') as f:
    json.dump([1,'abc'],f)
```

**反序列化**

```python
#从文件读json格式字符串反序列化
with open('test.json',mode='rt',encoding='utf-8') as f:
    json_res = f.read()
    rejson_res = json.loads(json_res)
    print(rejson_res,type(rejson_res)) #[1, 'abc'] <class 'list'>


#从文件读json格式字符串反序列化的简洁方法
with open('test.json',mode='rt',encoding='utf-8') as f:
   rejson_res = json.load(f)
   print(rejson_res, type(rejson_res))
```

**补充**

```python
json验证: json格式兼容的是所有语言通用的类型，集合类型几乎只有python有
json.dumps({1,2,3,4,5})  #TypeError: Object of type set is not JSON serializable
```





## Json格式

```python
'''
这些都是错的 都不是json格式
1. 'True'  
2. '[1,'str'] 
'''
l = json.loads('[1,"str",true]') #这是正确的
```



![image-20200805204707557](https://gitee.com/chzarles/images/raw/master/imgs/image-20200805204707557.png)





## 补充：Json.loads()参数可以是byte

```python
# 在python解释器2.7与3.6之后都可以json.loads(bytes类型)，但唯独3.5不可以

>>> import json
>>> json.loads(b'{"a":111}')
>>> Traceback (most recent call last):
>>> File "<stdin>", line 1, in <module>
>>> File "/Users/linhaifeng/anaconda3/lib/python3.5/json/__init__.py", line 312, in loads
>>> s.__class__.__name__))
>>> TypeError: the JSON object must be str, not 'bytes'
```



## [Json效率引出的编程思想](https://www.cnblogs.com/linhaifeng/articles/6384466.html#_label6)

```python
# 一.什么是猴子补丁?

      猴子补丁的核心就是用自己的代码替换所用模块的源代码，详细地如下

　　1，这个词原来为Guerrilla Patch，杂牌军、游击队，说明这部分不是原装的，在英文里guerilla发音和gorllia(猩猩)相似，再后来就写了monkey(猴子)。
　　2，还有一种解释是说由于这种方式将原来的代码弄乱了(messing with it)，在英文里叫monkeying about(顽皮的)，所以叫做Monkey Patch。


# 二. 猴子补丁的功能(一切皆对象)

　　1.拥有在模块运行时替换的功能, 例如: 一个函数对象赋值给另外一个函数对象(把函数原本的执行的功能给替换了)
class Monkey:
    def hello(self):
        print('hello')

    def world(self):
        print('world')


def other_func():
    print("from other_func")


monkey = Monkey()
monkey.hello = monkey.world
monkey.hello()
monkey.world = other_func
monkey.world()

# 三.monkey patch的应用场景

如果我们的程序中已经基于json模块编写了大量代码了，发现有一个模块ujson比它性能更高，
但用法一样，我们肯定不会想所有的代码都换成ujson.dumps或者ujson.loads,那我们可能
会想到这么做
import ujson as json，但是这么做的需要每个文件都重新导入一下，维护成本依然很高
此时我们就可以用到猴子补丁了
只需要在入口处加上
, 只需要在入口加上:

import json
import ujson

def monkey_patch_json():
    json.__name__ = 'ujson' #可加可不加
    json.dumps = ujson.dumps
    json.loads = ujson.loads

monkey_patch_json() # 之所以在入口处加，是因为模块在导入一次后，后续的导入便直接引用第一次的成果

#其实这种场景也比较多, 比如我们引用团队通用库里的一个模块, 又想丰富模块的功能, 除了继承之外也可以考虑用Monkey
Patch.采用猴子补丁之后，如果发现ujson不符合预期，那也可以快速撤掉补丁。个人感觉Monkey
Patch带了便利的同时也有搞乱源代码的风险!
猴子补丁与ujson

```

