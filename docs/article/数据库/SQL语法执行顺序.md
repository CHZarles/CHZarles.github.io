---
title: SQL语法执行顺序[转载]
comments: true
copyright: false
date: 2020-03-13 14:02:46
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/171198/pexels-photo-171198.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---



### [最最关键的一点：SQL 的语法并不按照语法顺序执行](http://dmcoders.com/2018/03/12/sql/)

SQL 语句有一个让大部分人都感到困惑的特性，就是：SQL 语句的执行顺序跟其语句的语法顺序并不一致。SQL 语句的语法顺序是：

<!--more-->

```
SELECT[DISTINCT]

FROM

JOIN

ON

WHERE

GROUP BY

HAVING

UNION

ORDER BY
```

为了方便理解，上面并没有把所有的 SQL 语法结构都列出来，但是已经足以说明 SQL 语句的语法顺序和其执行顺序完全不一样，就以上述语句为例，其执行顺序为：

```
FROM

JOIN

ON

WHERE

GROUP BY

AVG,SUM....

HAVING

SELECT

DISTINCT

UNION

ORDER BY

LIMIT/TOP
```

关于 SQL 语句的执行顺序的提示：

1、 FROM 才是 SQL 语句执行的第一步，并非 SELECT 。数据库在执行 SQL 语句的第一步是将数据从硬盘加载到数据缓冲区中，以便对这些数据进行操作。

2、 SELECT 是在大部分语句执行了之后才执行的，严格的说是在 FROM 和 GROUP BY 之后执行的。理解这一点是非常重要的，这就是你不能在 WHERE 中使用在 SELECT 中设定别名的字段作为判断条件的原因。

3、 无论在语法上还是在执行顺序上， UNION 总是排在在 ORDER BY 之前。很多人认为每个 UNION 段都能使用 ORDER BY 排序，但是根据 SQL 语言标准和各个数据库 SQL 的执行差异来看，这并不是真的。尽管某些数据库允许 SQL 语句对子查询（subqueries）或者派生表（derived tables）进行排序，但是这并不说明这个排序在 UNION 操作过后仍保持排序后的顺序。

4、并非所有的数据库对 SQL 语句使用相同的解析方式。如 MySQL、PostgreSQL和 SQLite 中就不会按照上面第二点中所说的方式执行。）







## [SQL语句的执行顺序](https://www.php.cn/sql/421993.html)

MySQL的语句一共分为11步，如下图所标注的那样，最先执行的总是FROM操作，最后执行的是LIMIT操作。其中每一个操作都会产生一张虚拟的表，这个虚拟的表作为一个处理的输入，只是这些虚拟的表对用户来说是透明的，但是只有最后一个虚拟的表才会被作为结果返回。如果没有在语句中指定某一个子句，那么将会跳过相应的步骤。

![undefined](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0ly1ge5qd6k12nj307j070weq.jpg)

**下面我们来具体分析一下查询处理的每一个阶段**

FORM: 对FROM的左边的表和右边的表计算笛卡尔积。产生虚表VT1

ON: 对虚表VT1进行ON筛选，只有那些符合<join-condition>的行才会被记录在虚表VT2中。

JOIN： 如果指定了OUTER JOIN（比如left join、 right join），那么保留表中未匹配的行就会作为外部行添加到虚拟表VT2中，产生虚拟表VT3, rug from子句中包含两个以上的表的话，那么就会对上一个join连接产生的结果VT3和下一个表重复执行步骤1~3这三个步骤，一直到处理完所有的表为止

WHERE： 对虚拟表VT3进行WHERE条件过滤。只有符合<where-condition>的记录才会被插入到虚拟表VT4中。

GROUP BY: 根据group by子句中的列，对VT4中的记录进行分组操作，产生VT5.

CUBE | ROLLUP: 对表VT5进行cube或者rollup操作，产生表VT6.

HAVING： 对虚拟表VT6应用having过滤，只有符合<having-condition>的记录才会被 插入到虚拟表VT7中。

SELECT： 执行select操作，选择指定的列，插入到虚拟表VT8中。

DISTINCT： 对VT8中的记录进行去重。产生虚拟表VT9.

ORDER BY: 将虚拟表VT9中的记录按照<order_by_list>进行排序操作，产生虚拟表VT10.

LIMIT：取出指定行的记录，产生虚拟表VT11, 并将结果返回。









