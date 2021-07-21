---
title: MYSQL-约束
comments: true
copyright: false
date: 2020-03-12 17:43:24
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/1170601/pexels-photo-1170601.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---





## SQL约束

SQL 约束用于规定表中的数据规则。

如果存在违反约束的数据行为，行为会被约束终止。

约束可以在创建表时规定（通过 CREATE TABLE 语句），或者在表创建之后规定（通过 ALTER TABLE 语句）



对约束的的修改是对表的结构的修改

<!--more-->



## 约束类型

- **NOT NULL** - 指示某列不能存储 NULL 值。
- **UNIQUE** - 保证某列的每行必须有唯一的值。
- **PRIMARY KEY** - NOT NULL 和 UNIQUE 的结合。确保某列（或两个列多个列的结合）有唯一标识，有助于更容易更快速地找到表中的一个特定的记录。
- **FOREIGN KEY** - 保证一个表中的数据匹配另一个表中的值的参照完整性。
- **CHECK** - 保证列中的值符合指定的条件。
- **DEFAULT** - 规定没有给列赋值时的默认值。
- **AUTO INCREMENT 字段** 补充





## MYSQL查看约束语法

```tcl
*方法一
SHOW CREATE TABLE 表名;
*方法二
SELECT * FROM information_schema.`TABLE_CONSTRAINTS`where table_name = 表名;
```



### SQL CREATE TABLE + CONSTRAINT 语法

```tcl
CREATE TABLE [IF NOT EXISTS] 表名 
(
	列名 列类型 [<完整性约束>[,<完整性约束>...]],
	列名 列类型 [<完整性约束>[,<完整性约束>...]],
	...
	列名 列类型 [<完整性约束>[,<完整性约束>...]]
);
```



## SQL NOT NULL 约束

```
*创建表时定义
CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255) NOT NULL,
    Age int
);

*添加NOT NULL约束
ALTER TABLE Persons
MODIFY Age int NOT NULL;

*删除NOT NULL约束
ALTER TABLE Persons
MODIFY Age int NULL;
```



## SQL UNIQUE 约束

```tcl
*创建表时定义
CREATE TABLE Persons
(
	P_Id int NOT NULL,
	LastName varchar(255) NOT NULL,
	City varchar(255),
	[UNIQUE(P_ID)]
	[CONSTRAINT 约束名 UNIQUE (P_Id,LastName[,列名])]
)

*增加约束
ALTER TABLE Persons
ADD UNIQUE (P_Id)

*增加约束
ALTER TABLE Persons
ADD CONSTRAINT uc_PersonID UNIQUE (P_Id,LastName)

*删除约束
ALTER TABLE Persons
DROP INDEX 键名/行名
```



## SQL PRIMARY KEY 约束

**每个表只能有一个主键**

```tcl
CREATE TABLE Persons
(
	P_Id int NOT NULL,
	LastName varchar(255) NOT NULL,
	//二选一
	[PRIMARY KEY (P_Id)]
	[CONSTRAINT <键名> PRIMARY KEY (P_Id,LastName)]//键名好像怎么写都无所谓
)

*插入主键
ALTER TABLE Persons
ADD PRIMARY KEY (P_Id)

ALTER TABLE Persons
ADD CONSTRAINT <键名> PRIMARY KEY (P_Id,LastName)//键名好像怎么写都无所谓

*删除主键
ALTER TABLE Persons
DROP PRIMARY KEY
```



## SQL CHECK 约束

```tcl
CREATE TABLE Persons
(
	P_Id int NOT NULL,
	City varchar(255),
	//二选一
	[CHECK (P_Id>0)]
	[CONSTRAINT 键名 CHECK (P_Id>0 AND City='Sandnes')]
)

*添加
ALTER TABLE Persons
ADD CHECK (P_Id>0)

ALTER TABLE Persons
ADD CONSTRAINT 键名 CHECK (P_Id>0 AND City='Sandnes')

*删除
ALTER TABLE Persons
DROP CHECK 键名
```





## SQL DEFAULT 约束

```tcl
CREATE TABLE Orders
(
    O_Id int NOT NULL,
    OrderNo int NOT NULL,
    P_Id int,
    OrderDate date DEFAULT 值/有返回值函数
)


*添加
ALTER TABLE Persons
ALTER City SET DEFAULT 'SANDNES'

*删除
ALTER TABLE Persons
ALTER City DROP DEFAULT
```



## AUTO INCREMENT 字段

```tcl
*类型是INT才行
CREATE TABLE Persons
(
	ID int NOT NULL AUTO_INCREMENT,
	LastName varchar(255) NOT NULL,
	FirstName varchar(255),
	Address varchar(255),
	City varchar(255),
	PRIMARY KEY (ID)
)

ALTER TABLE Persons AUTO_INCREMENT=100
```

