---
title: MYSQL-DQL笔记
comments: true
copyright: false
date: 2020-03-12 10:33:54
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/330771/pexels-photo-330771.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---









## 前言

**DQL语句不会改变表格结果，只是打印出信息会因表达式而改变**



**示例表格**：

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr195d5qfj30ij0a679g.jpg)



## 单表操作-列控制

```tcl
*查询所有列
 SELECT * FROM 表名;

*查询指定列
 SELECT [*] 列1 [,列2,...列N] FROM 表明;
 
*完全重复的记录只一次
  当查询结果中的多行记录一模一样时，只显示一行。一般查询所有列时很少会有这种情况，但只查询一列(或几列)时，	   这总可能就大了!
  SELECT DISTINCT * | 列1 [, 列2, ... 列N] FROM 表名;

*列运算
 **数量类型的列可以做 加、减、乘、除 运算
  SELECT sal*1.5 FROM emp;
  SELECT sal+corm FROM emp;
  SELECT * , sal*1.5 FROM emp;

 **字符串类型可以做连接运算
  SELECT CONCAT('工资:', sal [,字符串,...]) FROM emp;
  
 **转换NULL值
  有时需要把NULL转换成其它值，例如com+1000时， 如果com列存在NULL值，那么NULL+1000还是     NULL,而我们这时希望把NULL当成0来运算。  
  SELECT IFNULL(corm, 0)+1000 FROM emp;
  --> IFNULL (comm, 0): 如果Ccomm中存在NULL值，那么当成0来运算。
  
 **给列起别名
  你也许已经注意到了，当使用列运算后，查询出的结果集中的列名称很不好看，这时我们需要给列名起个   别名，这样在结果集中列名就显示别名了
  SELECT IFNULL(comm， 0)+1000 AS 奖金 FROM emp;
  --> 其中As可以省略
```





## 单表操作-条件控制

```tcl
*条件查询
 与前面介绍的UPDATE和DELETE语句-样，SELECT语句 也可以使用WHERE子句来控制记录。
 SELECT empno, ename, sal,comm FROM emp WHERE sal > 10000 AND comm IS NOT NULL;
 SELECT empno, ename,sal FROM emp WHERE sal BETWEEN 20000 AND 30000;
 SELECT empno, ename,job FROM emp WHERE job IN ('经理'， '董事长');
 
*模糊查询
 当你想查询姓张，并且姓名一共两个字的员工时，这时就可以使用模糊查询
 SELECT 米FROM emp WHERE ename LIKE '张_ ';
 -->模糊查询需要使用运算符: LIKE, '_'其中匹配一个任意字符，注意，只匹配一个字符而不是多个。
 -->上面语句查询的是姓张，名字由两个字组成的员工。
 
 SELECT * FROM emp WHERE ename LIKE '___'; /*姓名由3个字组成的员工*/
 如果我们想查询姓张，名字几个字都可以的员工时就要使用“%”了。
 
 SELECT * FROM emp. WHERE ename LIKE '张%';
 -->其中匹配0~N个任意字符，所以上面语句查询的是姓张的所有员工。
 
 SELECT * FROM emp WHERE ename LIKE '%阿%';
 -->千万不要认为上面语句是在查询姓名中间带有阿字的员工，因为匹配0~N个字符，所以姓名以阿开头和结尾的员工也 都会查询到。
 
 SELECT * FROM emp WHERE ename LIKE '%';
 -->这个条件等同与不存在，但如果姓名为NULL的查询不出来!
 
*模糊搜索补充
 如果目标字段包含'%'或'_',要用语法 ESCAPE'<转义字符>'。
 比如，我们要搜索一个字符串"g_",如果直接 LIKE "g_"，那么"_"的作用就是通配符，而不是字符
 结果，我们会查到比如"ga","gb","gc",而不是我们需要的 "g_".
 用LIKE'gs_' ESCAPE 'S'   
 'S'表示特殊用法标志
```



## 单表操作-排序

```tcl
*升序
SELECT * FROM emp ORDER BY sal ASC
-->按sa1排序，升序
-->其中ASC是可以省略的

*降序
SELECT * FROM emp ORDER BY comm DESC;
-->按comm排序，降序!
-->其中DESC不能省略

*使用多列作为排序条件 前面的条件优先级高
SELECT * FROM  emp ORDER BY sal ASC , comm DESC: 
```





## 单表操作-聚合函数

```tcl
*COUNT
 SELECT COUNT(*) FROM emp;
 -->计算emp表中所有列都不为NULL的记录的行数
 SELECT COUNT (comm) FROM emp;
 -->计算emp表中cormm列不为NULL的记录的行数
 
*MAX
 SELECT MAX(sa1) FROM emp;
 -->查询最高工资
 
*MIN
 SELECT MIN(sal) FROM emp;
 -->查询最低工资
 
*SUM
 SELECT SUM(sal) FROM emp;
 -->查询工资和

*AVG
 SELECT AVG(sal) FROM emp;
 -->查询平均工资
```





## 单表操作-分组

**将查询结果按属性分组，一般按 有信息重复的列 分组才有意义**。

**一般分组配合聚合函数使用(例），除了分组列是列名，其他都是聚合函数**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr4eiode8j30eo05vab9.jpg)

```tcl
*分组查询
 分组查询是把记录使用某一列进行 分组，然后查询组信息。
 例如:查看所有部门的记录数。
 SELECT deptno，COUNT (*). FROM emp GROUP BY deptno;
 -->使用deptno分组，查询部门编号和每个部门的记录数
 SELECT job, MAX(SAL). FROM emp GROUP BY job;
 -->使用job分组，查询每种工作的最高工资
 
*组条件
 以部门分组，查询每组记录数。条件为记录数大于3
 SELECT deptno, COUNT (*) FROM emp GROUP BY deptno HAVING  COUNT(*)>3;
```





## 单表操作-语句顺序

```tcl
*语法顺序是如此，执行顺序也是如此
 SELECT  FROM  WHERE  GROUP BY   HAVING   ORDER BY
```

**例：查询 工资大于15000的人员的人数 超过两个的部门（没有排序）。**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gcr4qwmbs5j30ne04nt9k.jpg)





## 单表操作-方言LIMIT

**用于实现分页查询**

```tcl
LIMIT用来限定查询结果的起始行，以及总行数。
例如:查询起始行为第5行，一共查询3行记录
SELECT * FROM emp LIMIT 4，3;
-->其中4表示从第5行开始，其中3表示一共查询3行。 即第5、6、7行记录。

起始页: (当前页-1) * 每页记录数
```

