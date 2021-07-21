---
title: MYSQL-索引
comments: true
copyright: false
date: 2020-03-12 16:40:25
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/1707727/pexels-photo-1707727.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---





# 聚焦索引和非聚集索引



一个数据库表建立索引会建立索引文文件，索引文件除了有查询要用到的搜索码(表中某些属性列的值)，还要有指示位置的信息（指针|具有指向性的值）。

​	在索引文件找到一个值就是找到一个地址（指向地址的值）

<!--more-->

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr8mdp9rrj30ou0cc78b.jpg)

# 聚集索引



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr8qc1x32j30kg0bw40e.jpg)





# 非聚集索引

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr8p3ikp6j30mf0ckacv.jpg)



# 说明

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr8sltszej30nl0by41w.jpg)





# 语法

```tcl
*建立索引
CREATE [UNIQUE] [CLUSTER] INDEX <索引名> ON <表名>(<列名>[<次序>][,<列名>[<次序>]]...);

-->次序{ASC|DESC} 默认ASC
-->CLUSTER表明建立聚集索引
-->UNIQUE表明此索引每一个索引值只对应唯一的数据记录
```



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr953c3kdj30cd054gmb.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr95iotakj30co06kjsc.jpg)

```tcl
*索引删除
 ALTER TABLE table_name DROP INDEX index_name
```



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcra928a4kj30e50dvgut.jpg)