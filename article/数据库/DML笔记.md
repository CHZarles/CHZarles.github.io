---
title: MYSQL-DML笔记
comments: true
copyright: false
date: 2020-03-12 09:09:50
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/1342460/pexels-photo-1342460.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260
toc: true
---





## 前言

DML (数据操作语言，它是对表记录的操作(增、删、改)! )

```tex
查询语句: SELECT * FROM 表名
```

<!--more-->



## 插入数据

```tcl
//插入所有列
 INTERT INTO 表名(列名1,列名2,... ) VALUES(列值1,列值2,...);
	>在表名后给出要插入的列名，其他没有指定的列等同与插入nu11值。
	>在VALUES后给出列值，值的顺序和个数必须与前面指定的列对应。
	
//插入部分列
 INTERT INTO 表名 VALUES (列值1，列值2)
	>没有给出要插入的列，那么表示插入所有列。
	>值的个数必须是该表列的个数。
	>值的顺序，必须与表创建时给出的列的顺序相同。
```

**注意：数据库中所有的字符串类型，必须使用单引号，不能用双引号!  日期类型也要用单引号!**





## 修改数据

```tcl
//根据逻辑表达式修改对应行语句
 update 表名 set 列名1=列值1 [,列名2=列值2....] [where 逻辑表达式]

>运算符:=、!=、<>、>,<、>=、<=、BETWEEN...AND、 IN(. ..)、 IS NULL、 NOT、 OR、 AND

 WHERE age BETWEEM 20 AND 40  <=>  WHERE age>=20 AND age<=40
 WHERE name='张三' or name='李四' <=> WHERE name IN ('张三'，'李四')
 WHERE 列值 = NULL  永远返回 False,改用  WHERE 列值 IS NULL
 IS NULL 相对的是 IS NOT NULL
```





## 删除数据

```tcl
//不加where就整个表都清空了
 DELETE FROM 表名 [WHERE 条件表达式]

//TRUNCATE是DDL语句，它是先删除drop该表，再create该表。 而且无法回滚! ! !
 TRUNCATE TABLE 表名: 
```

