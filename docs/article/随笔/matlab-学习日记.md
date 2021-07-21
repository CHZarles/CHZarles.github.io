---
title: matlab 学习日记 10.17
date: 2019-10-17 20:59:33
tags: matlab
photo: 
categories: 
	- 随笔

cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## 开发环境介绍

我用的是中文版的matlab，左边的文件区可以选择要打开的文件，编辑器就是文本编辑器用于写代码再运行，命令行窗口就是边输入边运行，工作区可以显示代码工作时各种变量及其对应的值。

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g81hz66b5lj31hc0spdlk.jpg)

## 基本数据类型

1.数字

2.字符和字符串

3.矩阵

4.元胞数组

<!--more-->
5.结构体

## 基本应用

小细节

```matlab
%%清除环境变量和命令
clear all
%清空命令行窗口
clc         
%%变量名以字母开头
s = 4
```

字符串

```matlab
%%字符与字符串
s = 'a'
abs(s)
char(65)
num2str(65) %把数字变成了两个字符串

str = 'charles blog ' %字符串和字符是不区分的

length(str)

%有很多字符串的函数，可以通过查看帮助文档来查看，文末提供了matlab中文手册的下载地址
```

矩阵(数组)

```matlab
%矩阵的定义和赋值
A = [1 2 3 ; 4 5 6 ; 3 2 7] %3*3矩阵，直接赋值就行，不用指定类型
B = A' %矩阵转置
C = A(:) %矩阵A转化为列向量
D = inv(A) %取逆矩阵
A * D %转单位矩阵

E = zeros(10,5,3) %10*5*3的三维数组
E(:,:,1) = rand(10,5) %给第一层二维数组赋随机值
E(:,:,2) = randi(5,10,5) %给第一层二维数组赋随机值
E(:,:,3) = randn(10,5) %赋值

A = 1:2:9         % 矩阵 [ 1 3 5 7 9 ] 有点像python
B = repmat(A,3,1) %复制A到B 把A看成一个数，复制3行1列
D = ones(2,4)      %两行四列个1的矩阵

%矩阵的运算
A = [1 2 3 4 ; 5 6 7 8]
B = [1 1 2 2 ; 2 2 1 1]
C = A + B
D = A - B
E = A * B' %矩阵的乘法运算
F = A.* B  %对应位置相乘
G = A / B  %等价 G * B = A , G * pinv(B) * B = A * pinv(B) 
H = A./ B  %对应位置相除
pinv(B) %求B的伪逆矩阵

%矩阵取下标
A = magic(5) %定义一个5*5的魔方数组
B = A(2,3)
C = A( :,3 ) %取第三列
D = A( 2,: ) %取第二行
[m,n] = find(A > 20) %找符合要求的以数组形式返回下标

```

联合体（我瞎编的名字）

```matlab
%%元胞数组(就是多维数组的另一种表现形式)
A = cell(1,6)
A{2} = eye(3)
A{5} = magic(5)
B = A{5}

%%结构体
books = struct('name',{{'Machine Learning','Data Mining'}},'price',[30 40]) %这个结构体包含name和price两个数组
books.name
books.name(1) %返回一个元胞数组
books.name{1} %返回一个字符串的形式

```

## 资源下载

[Matlab使用中文手册.pdf](/download/MATLAB实用中文手册.pdf)

