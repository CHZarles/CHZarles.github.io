---
title: 项目日志-loggiing模块
toc: true
copyright: true
comments: true
date: 2020-08-08 23:37:46
keyword:
tags:
categories: python开发相关
top:
---



## logging模块的简单入门

**日志是用来输出程序运行信息的**

```python
# 日志模块的使用
import logging


#1.日志配置
# 简单地自定义日志格式，这是模块提供的方法，记住api就好
logging.basicConfig(

    #日志输出位置：1、终端 2、文件
    #filename='access.log', # 不指定，默认打印到终端

    # 2、日志格式 可以自定义 就用这个模板就好
    #asctime是时间格式
    #name是日志名(默认是root)
    #levelname 日志级别
    #module 发出日志所在的模块
    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s  ' ,

    # 3、时间格式
    datefmt='%Y-%m-%d %H:%M:%S %p',

    # 4、日志级别 低于level的就不打印
    # critical => 50
    # error => 40
    # warning => 30
    # info => 20
    # debug => 10
    level=30,
)
```

<!--more-->

```python
logging.debug('调试debug')
logging.info('消息info')
logging.warning('警告warn')
logging.error('错误error')
logging.critical('严重critical')

输出:
2020-08-08 23:47:05 PM - root - WARNING -day1-basic_variable:  警告warn  
2020-08-08 23:47:05 PM - root - ERROR -day1-basic_variable:  错误error  
2020-08-08 23:47:05 PM - root - CRITICAL -day1-basic_variable:  严重critical  
```





## 高级用法

之前讲的日志配置只是基本的配置信息
在项目中，我们通常会将各种日志信息集成在字典里 ，要用的时候加载进去。



### 格式化字符串表

日志中可能用到的格式化串如下（不用背，要定制的时候根据需求查这个表就行）

```python
#1.格式化字符串
%(name)s  Logger的名字
%(levelno)s  数字形式的日志级别|
%(levelname)s 文本形式的日志级别
%(pathname)s 调用日志输出函数的模块的完整路径名，可能没有
%(filename)s 调用日志输出函数的模块的文件名
%(module)s 调用日志输出函数的模块名
%(funcName)s 调用日志输出函数的函数名
%(lineno)d 调用日志输出函数的语句所在的代码行
%(created)f 当前时间，用UNIX标准的表示时间的浮 点数表示
%(relativeCreated)d 输出日志信息时的，自Logger创建以 来的毫秒数
%(asctime)s 字符串形式的当前时间。默认格式是 “2003-07-08 16:49:45,896”。逗号后面的是毫秒
%(thread)d 线程ID。可能没有
%(threadName)s 线程名。可能没有
%(process)d 进程ID。可能没有
%(message)s 用户输出的消息
```



### 定制format

```python
# setting.py
# 2、强调：其中的%(name)s为getlogger时指定的名字，其它的查上面的表
standard_format = '[%(asctime)s][%(threadName)s:%(thread)d][task_id:%(name)s][%(filename)s:%(lineno)d]' \
                  '[%(levelname)s][%(message)s]'

simple_format = '[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]%(message)s'

test_format = '%(asctime)s] %(message)s'
```



### 配置字典

```python
# setting.py
# 3、日志配置字典  重点是 formatters handlers loggers 其它记不记无所谓
LOGGING_DIC = {
    'version': 1, #版本信息 自己编辑
    'disable_existing_loggers': False,
    #formatters 也是一个字典 里面的key名(如 stabdard,simple) 是自己定义的名字
    'formatters': {
        'standard': {
            #这个key不能改
            'format': standard_format
        },
        'simple': {
            'format': simple_format
        },
        'test': {
            'format': test_format
        },
    },
    'filters': {},
    # 在handlers设置日志的接收者，不同的handler会将日志输出到不同的位置，里面的key是自定义的
    'handlers': {
        #打印到终端的日志
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',  # 打印到屏幕
            'formatter': 'simple'
        },
        #打印到文件的日志,收集info及以上的日志
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件,日志轮转
            'formatter': 'standard',
            # 可以定制日志文件路径
            # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # log文件的目录
            # LOG_PATH = os.path.join(BASE_DIR,'a1.log')
            'filename': 'a1.log',  # 日志文件
            'maxBytes': 1024*1024*5,  # 日志大小 5M 。
            'backupCount': 5, #最多轮转5个文件，最新的日志总是存在a1.log ，溢出的内容就往回退
            'encoding': 'utf-8',  # 日志文件的编码，再也不用担心中文log乱码了
        },
        'other': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',  # 保存到文件
            'formatter': 'test',
            'filename': 'a2.log',
            #这个位置可以设计文件路径 通常为 项目名/kog/a2.log
            # 这样写 oS. path. join(os。path. dirname ( os. path. dirname(__ file__ )),'log', 'a2. Log')
            'encoding': 'utf-8',
        },
    },
    # loggers是日志的产生者 产生的日志会传递给handler
    # loggers是给程序员调用的对象 上面的只是给logger的配置
    'loggers': {
        #logging.getLogger(__name__)拿到的logger配置
        'kkk': {
            'handlers': ['default', 'console'],  # 这里把上面定义的两个handler都加上，即log数据既写入文件又打印到屏幕
            'level': 'DEBUG', # loggers(第一层日志级别关限制)--->handlers(第二层日志级别关卡限制)
            'propagate': False,  # 了解了解就行 默认为True，向上（更高level的logger）传递，通常设置为False即可，否则会一份日志向上层层传递
        },
        'key都是可以自定义的': {
            'handlers': ['other',],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
```



## 获取日志对象应用实例

```python
 # 拿到日志的产生者 loggers 来产生日志
 # 先导入日志配置字典


import settings   #settings是刚才编辑日志配置字典的文件
# import logging.config as tmp_config
#logging是一个包 这个包的名称空间_init__.py里面的__all__[] 没有config,config是它的子包
#在执行这个语句的同时，logging也被导入了，因为python的找子模块机制就是，先检查__init__.py文件，再检查子文件夹
#                  这个过程中需要导入logging模块
# 另一种写法是: from logging import config

from logging import config,getLogger

# 加载字典的方法
config.dictConfig(settings.LOGGING_DIC)
# 获取日志产生者
logger1 = getLogger('kkk')

#产生日志
logger1.info('这是最最新一条日志')
```