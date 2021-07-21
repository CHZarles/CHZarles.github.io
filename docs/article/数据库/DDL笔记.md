---
title: MYSQL-DDL笔记
comments: true
copyright: false
date: 2020-03-11 17:20:55
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/1714202/pexels-photo-1714202.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---





## 前言

**此文仅记录一些简单的语法**

**DDL 都是对数据库，数据表的结构的操作**



## DATABASE操作

```tcl
1.数据库
* 查看所有数据库: SHOW DATABASES
* 切换数据库: USE 数据库名
* 创建数据库: CREATE DATABASE [IF NOT EXISTS] 数据库名 [CHARSET=utf8]
* 删除数据库: DROP DATABASE [IF EXISTS] 数据库名
* 修改数据库编码: ALTER DATABASE 数据库名 CHARACTER SET utf8
```

<!--more-->

## 数据类型

**主要数据类型**

```
int: 整型
double: 浮点型，例如double (5,2)表示最多5位，其中必须有2位小数， 即最大值为999.99;
decimal: 浮点型，在表单钱方面使用该类型，因为不会出现精度缺失问题
char:固定长度字符串类型: char(255), 数据的长度不足指定长度，补足到指定长度!
varchar: 可变长度字符串类型: varchar (65535)， 不自动补长，但会用一定空间记录数据长度
text (clob):字符串类型:
blob: 字节类型:
date: 日期类型，格式为: yyy-MM-dd;
time:时间类型，格式为: hh:mm:ss
cimestamp: 时间戳类型
```



**补充：**

字符串类型

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcq5jyrszvj30jz0csdn1.jpg)



blob类型

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcq5pabd06j30n607oq7u.jpg)



## 表操作



```tcl
*创建表
CREATE TABLE [IF NOT EXISTS] 表名 
(
	列名 列类型，
	列名 列类型，
	...
	列名 列类型 
);

*查看当前数据车中所有表名称: SHOW TABLES;
*查看指定表的创建语句: SHOW CREATE TABLE表名(了解);
*查看表结构: DESC 表名;
*删除表: DROP TABLE 表名;
*修改表: 
	ALTER TABLE 表名 + 的后续语句
	>修改之添加列:
	>ALTER TABLE 表名 ADD (
		列名 列类型,
		列名 列类型,
		...
	);
	
	>修改之修改“列类型”(如果被修改的列已存在数据，那么新的数据可能会影响已经存在的数据)
				   ALTER TABLE 表名 MODIFY 列名 数据类型;
	>修改之修改列名: ALTER TABLE 表名 CHANGE 原列名 新列名 列类型;
	>修改之删除列: ALTER TABLE 表名 DROP 列名;
	>修改表名称: ALTER TABLE 原表名RENAME TO 新表名;
```



