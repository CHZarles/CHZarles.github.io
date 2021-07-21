---
title: MYSQL-多表查询
comments: true
copyright: false
date: 2020-03-12 19:51:26
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/1670977/pexels-photo-1670977.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---







## 集合操作

### UNION

> UNION 操作符用于合并两个或多个 SELECT 语句的结果集。
>
> 
>
> 请注意，UNION 内部的每个 SELECT 语句必须拥有相同数量的列。列也必须拥有相似的数据类型。同时，每个 SELECT 语句中的列的顺序必须相同。
>
> 
>
> 对于union的前部分查询和后部分查询不能有`GROUP BY,ORDER BY`等字段,只有是在整个的最后才能有`GROUP BY,ORDER BY`等字段!



```
SELECT column_name(s) FROM  table1
UNION [ALL]
SELECT column_name(s) FROM  table2
```



## EXCEPT

> MYSQL 不支持这个操作 ，`NOT IN` 可以实现这个操作,`JOIN`也可操作实现
>



## INTERSECT

> MYSQL 不支持这个操作,利用`JOIN`操作实现
>




## 笛卡尔积

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcregbx7hbj30s80ewdhl.jpg)



<!--more-->

## 等值连接

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcrf6nd0ckj30r00d9wg3.jpg)







## 自然连接

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcria9zmygj30vg0e00ua.jpg)

```tcl
*mysql语句，如果两个表只有一个属性相同
select * from  s natural join sc;
```





## 自身连接

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcrii6frlgj30vw0extat.jpg)







## JOIN 操作



**SQL join 用于把来自两个或多个表的行结合起来。**



> 首先，连接的结果可以在逻辑上看作是由SELECT语句指定的列组成的新表。
>
> 左连接与右连接的左右指的是以两张表中的哪一张为基准，它们都是外连接。
>
> 外连接就好像是为非基准表添加了一行全为空值的万能行，用来与基准表中找不到匹配的行进行匹配。假设两个没有空值的表进行左连接，左表是基准表，左表的所有行都出现在结果中，右表则可能因为无法与基准表匹配而出现是空值的字段。
>
> ​																									





![undefined](http://ww1.sinaimg.cn/large/006eb5E0gy1gcrkdllby4j30qu0l4wix.jpg)



手册:[1.SQL连接操作](https://www.runoob.com/sql/sql-join.html)

​         [2.MYSQL连接操作](https://www.runoob.com/mysql/mysql-join.html)

## SQL JOIN 中 on 与 where 的区别

![undefined](http://ww1.sinaimg.cn/large/006eb5E0gy1gcrk97xv0aj311y0kgt9c.jpg)

- **left join** : 左连接，返回左表中所有的记录以及右表中连接字段相等的记录。
- **right join** : 右连接，返回右表中所有的记录以及左表中连接字段相等的记录。
- **inner join** : 内连接，又叫等值连接，只返回两个表中连接字段相等的行。
- **full join** : 外连接，返回两个表中的行：left join + right join。
- **cross join** : 结果是笛卡尔积，就是第一个表的行数乘以第二个表的行数。

### 关键字 on

数据库在通过连接两张或多张表来返回记录时，都会生成一张中间的临时表，然后再将这张临时表返回给用户。

在使用 **left jion** 时，**on** 和 **where** 条件的区别如下：

- 1、 **on** 条件是在生成临时表时使用的条件，它不管 **on** 中的条件是否为真，都会返回左边表中的记录。
- 2、**where** 条件是在临时表生成好后，再对临时表进行过滤的条件。这时已经没有 **left join** 的含义（必须返回左边表的记录）了，条件不为真的就全部过滤掉。

假设有两张表：

**表1：tab2**

| id   | size |
| ---- | ---- |
| 1    | 10   |
| 2    | 20   |
| 3    | 30   |

**表2：tab2**

| size | name |
| ---- | ---- |
| 10   | AAA  |
| 20   | BBB  |
| 20   | CCC  |

两条 SQL:

```
select * form tab1 left join tab2 on (tab1.size = tab2.size) where tab2.name='AAA'

select * form tab1 left join tab2 on (tab1.size = tab2.size and tab2.name='AAA')
```

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcrkas02woj30j80c6dgd.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcrkb67thuj30jc074mxg.jpg)

其实以上结果的关键原因就是 **left join、right join、full join** 的特殊性，不管 **on** 上的条件是否为真都会返回 **left** 或 **right** 表中的记录，**full** 则具有 **left** 和 **right** 的特性的并集。 而 **inner jion** 没这个特殊性，则条件放在 **on** 中和 **where** 中，返回的结果集是相同的。

[原文链接](https://www.runoob.com/w3cnote/sql-join-the-different-of-on-and-where.html)