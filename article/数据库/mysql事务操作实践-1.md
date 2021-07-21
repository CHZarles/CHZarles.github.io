---
title: mysql事务操作实践(1)
comments: true
copyright: false
date: 2020-04-28 15:35:50
tags: 
	- 数据库
	- 事务
categories: 数据库
photo:
top:
cover: https://t9.baidu.com/it/u=583874135,70653437&fm=79&app=86&size=h300&n=0&g=4n&f=jpeg?sec=1588666223&t=3caf3ec50636b415e6ffdf9f57d0c311
toc: true
---



## 理论准备（未全部验证）

### MYSQL的COMMIT和ROLLBACK

从功能上划分，SQL 语言可以分为DDL,DML和DCL三大类。

1. DDL(Data Definition Language)

数据定义语言，用于定义和管理 SQL [数据库](https://www.2cto.com/database/)中的所有对象的语言 ;

CREATE---创建表

ALTER---修改表

DROP---删除表

2. DML(Data Manipulation Language)

数据操纵语言，SQL中处理数据等操作统称为数据操纵语言 ;

INSERT---数据的插入

DELETE---数据的删除

UPDATE---数据的修改

SELECT---数据的查询



<!--more-->



3. DCL(Data Control Language)

数据控制语言，用来授予或回收访问数据库的某种特权，并控制 数据库操纵事务发生的时间及效果，对数据库实行监视等;

GRANT--- 授权。

ROLLBACK---回滚。

COMMIT--- 提交。

4. 提交数据有三种类型：显式提交、隐式提交及自动提交。

下面分 别说明这三种类型。

(1) 显式提交

用 COMMIT 命令直接完成的提交为显式提交。

(2) 隐式提交

用 SQL 命令间接完成的提交为隐式提交。这些命令是：

ALTER ， AUDIT ， COMMENT ， CONNECT ， CREATE ， DISCONNECT ， DROP ， EXIT ， GRANT ， NOAUDIT ， QUIT ， REVOKE ， RENAME 。

(3) 自动提交

若把 AUTOCOMMIT 设置为 ON ，则在插入、修改、删除语句执行后，[系统](https://www.2cto.com/os/)将自动进行提交，这就是自动提交。其格式为： SQL>SET AUTOCOMMIT ON ;

COMMIT / ROLLBACK这两个命令用的时候要小心。 COMMIT / ROLLBACK 都是用在执行 DML语句(INSERT / DELETE / UPDATE / SELECT )之后的。DML 语句，执行完之后，处理的数据，都会放在回滚段中(除了 SELECT 语句)，等待用户进行提交(COMMIT)或者回滚 (ROLLBACK)，当用户执行 COMMIT / ROLLBACK后，放在回滚段中的数据就会被删除。

(SELECT 语句执行后，数据都存在共享池。提供给其他人查询相同的数据时，直接在共享池中提取，不用再去数据库中提取，提高了数据查询的速度。)

所有的 DML 语句都是要显式提交的，也就是说要在执行完DML语句之后，执行 COMMIT 。而其他的诸如 DDL 语句的，都是隐式提交的。也就是说，在运行那些非 DML 语句后，数据库已经进行了隐式提交，例如 CREATE TABLE，在运行脚本后，表已经建好了，并不在需要你再进行显式提交。

在提交事务(commit)之前可以用rollback回滚事务。

\#commit、rollback用来确保数据库有足够的剩余空间;

\#commi、rollback只能用于DML操作，即insert、update、delet;

\#rollback操作撤销上一个commit、rollback之后的事务。

create table test

(

PROD_ID varchar(10) not null,

PROD_DESC varchar(25) null,

COST decimal(6,2) null

);

\#禁止自动提交

set autocommit=0;

\#设置事务特性,必须在所有事务开始前设置

\#set transaction read only; #设置事务只读

set transaction read write; #设置事务可读、写

\#开始一次事务

start transaction;

insert into test

values('4456','mr right',46.97);

commit; #位置1

insert into test

values('3345','mr wrong',54.90);

rollback; #回到位置1，(位置2);上次commit处

insert into test

values('1111','mr wan',89.76);

rollback; #回到位置2，上次rollback处

\#测试保存点savepoint

savepoint point1;

update test

set PROD_ID=1;

rollback to point1; #回到保存点point1

release savepoint point1; #删除保存点

drop table test;



## Mark

1. rollback要在commit前执行才能rollback
2. mysql默认隐式提交(什么是隐私提交？待研究)










