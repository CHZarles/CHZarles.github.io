---
title: MYSQL—DCL笔记
comments: true
copyright: false
date: 2020-03-12 10:08:27
tags:
	- 数据库
categories:
	- 数据库
photo:
top:
cover: https://images.pexels.com/photos/1089440/pexels-photo-1089440.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500
toc: true
---







## 前言

**通常一个项目创建一个用户!一个项目对应的数据库只有一个!**

**这个用户只能对这个数据库有权限**



## 创建用户

```tcl
//创建用户
 CREATE USER 用户名@IP地址 IDENTIFIED BY '密码':
>用户只能在指定的IP地址上登录

 CREATE USER 用户名@'%' IDENTIFIED BY '密码';
>用户可以在任意IP地址上登录
```

<!--more-->



## 给用户授权

```tcl
GRANT 权限1,..权限n ON 数据库.*  TO  用户名@IP地址
 >权限、用户、数据库
 >给用户分派在指定的数据库上的指定的权限
 >例如: GRANT CREATE, ALTER, DROP , INSERT, UPDATE, DELETE, SELECT ON mydb1.H TO user1@1ocalhost;
 	*给user1用户分派在mydb1数据库.上的create、alter. drop. insert、 update、 delete、 select权限

GRANT ALL ON 数据库.* TO用户名@IP地址:
 >给用户分派指定数据库.上的所有权限
```



## 撤销授权

```tcl
 REVOKE 权限1,...,权限n ON 数据库.* FROM 用户名@IP地址;
	>撤消指定用户在指定数据库上的指定权限
	>例如: REVOKE CREATE, ALTER, DROP ON mydb1.* FROM user1@1ocalhost;
		*撤消user1用户在mydb1数据库.上的create,alter,drop权限
```



## 查看权限

```tcl
SHOW GRANTS FOR 用户名@IP地址;
 >查看指定用户权限
```



## 删除用户

```tcl
DROP USER 用户名@IP地址
```

