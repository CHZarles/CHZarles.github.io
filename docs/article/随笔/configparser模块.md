---
title: 操作配置文件-configparser模块
toc: true
copyright: true
comments: true
date: 2020-08-06 07:02:41
keyword:
tags:
categories: python开发相关
top:
---



## Configparser模块和配置文件

```python

# configparser模块 用来加载某种格式的配置文件 通常以ini结尾 cfg结尾
# 这类配置文件的特征在于 这些结构的文件的结构 不在于文件后缀 
# 这些配置文件 兼容json格式
'''

[section1]
k1 = v1 #井号是注释符
k2:v2 ;分号也可以当注释符 这里和 k2 = v2 含义一样
k3 = v3
[section2]
k4 = v4
''

#mysqlserver.ini 部分截取
[client]
port=3306

[mysql]
no-beep
default-character-set=utf8

[mysqld]
character_set_server=utf8

'''
```

<!--more-->



## 配置文件读取方法

```python

# test.ini
[client]
port=3306
name=charles
website = http:/chzarles.gitee.io

[mysql]
default-character-set=utf8
power:root
readable:true

[mysqld]
character_set_server=utf8


#*.py
import configparser
# 取配置文件的配置项 : 先找到section 再找到option 最终定位到value

config = configparser.ConfigParser()
config.read('test.ini',encoding='utf-8') #一次性读入配置文件到内存 配置文件不会很大 不用担心爆内存
#没指定encode=utf-8报错了: UnicodeDecodeError: 'gbk' codec can't decode byte 0xaa in position 27: illegal multibyte sequence

# 1.获取所有sections 以列表形式返回
print(config.sections()) #['client', 'mysql', 'mysqld']

# 2.获取某一个section下的配置项 以列表形式返回
print(config.options('client')) #['port', 'name', 'website']

# 3.获取某一个section下的item 列表形式返回 列表内元素全是二元组
print(config.items('client'))
# [('port', '3306'), ('name', 'charles'), ('website', 'http:/chzarles.gitee.io')]

# 4.获取指定的配置项的值 例如:我想只获得 port 对应的值 ; get方法的返回值是字符串
res = config.get('client','port')
print(res,type(res)) #3306 <class 'str'>
# 4.1 直接获得int的一步到位的方法
res = config.getint('client','port')
print(res,type(res)) #3306 <class 'int'>

'''
类似方法还有
val2=config.getboolean('section1','is_admin')
val3=config.getfloat('section1','salary')
```



## 配置文件的修改方法（了解一下就行）

```python
# 修改配置文件的方法
# 了解一下就行 因为通常配置文件都是用户手动改的 不是程序运行过程改的
import configparser

config=configparser.ConfigParser()
config.read('a.cfg',encoding='utf-8')


#删除整个标题section2
config.remove_section('section2')

#删除标题section1下的某个k1和k2
config.remove_option('section1','k1')
config.remove_option('section1','k2')

#判断是否存在某个标题
print(config.has_section('section1'))

#判断标题section1下是否有user
print(config.has_option('section1',''))


#添加一个标题
config.add_section('egon')

#在标题egon下添加name=egon,age=18的配置
config.set('egon','name','egon')
config.set('egon','age',18) #报错,必须是字符串


#最后将修改的内容写入文件,完成最终的修改
config.write(open('a.cfg','w'))
```

## 基于上述方法添加一个ini文档

```python
import configparser

config = configparser.ConfigParser()
#section DEFAULT
config["DEFAULT"] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}

config['DEFAULT']['ForwardX11'] = 'yes'

#section [bitbucket.org]
config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'

# section [topsecret.server.com]
config['topsecret.server.com'] = {}
topsecret = config['topsecret.server.com'] #返回值是一个字典
topsecret['Host Port'] = '50022'  # mutates the parser
topsecret['ForwardX11'] = 'no'  # same here


#写入文件
with open('example.ini', 'w') as configfile:
    config.write(configfile)

```

输出结果

```ini
#example.ini
[DEFAULT]
serveraliveinterval = 45
compression = yes
compressionlevel = 9
forwardx11 = yes

[bitbucket.org]
user = hg

[topsecret.server.com]
host port = 50022
forwardx11 = no
```

