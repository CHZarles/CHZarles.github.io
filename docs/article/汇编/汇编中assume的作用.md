---
title: 汇编中assume的作用[转]
comments: true
copyright: false
date: 2020-05-24 19:13:09
tags:
categories: 汇编
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0ly1gf3s0oevd5j306k06kweh.jpg
toc: true
---





## 前言

为了防止原作者内容丢失，copy一份

[原帖链接](https://blog.csdn.net/qq_40627648/article/details/84063124)

<!--more-->

## assume 的作用是关联段名与段寄存器。

如果你在数据段中定义了变量名，比如：

```assembly
X DB 0
```


而你在代码中，需要直接使用这个变量名，比如：

```assembly
MOV AL,X
```


那么，汇编程序在汇编时，就会报告错误。

那么，汇编程序在汇编时，就会报告错误。

因为，mov指令中遇到 x 这个变量名时，汇编程序不知道它要用哪个段寄存器作为段地址。

**所以若要用变量名直接访问，或使用语句标号（比如你例子中的标号 start）就必须要在assume伪指令中将这些变量或标号所在段的段名，与段寄存器名关联，否则会出错。**

如果你不使用段中的变量名，可以不关联这个段的段名与寄存器。

**如果你访问变量时，都指定了段跨越前缀，关联也不是必须的。比如你可以用 `MOV AL, DS:X`访问变量 X**（我感觉这个代码有问题）



## 代码段中手动指定

这几天在看王爽大大的 汇编语言。对于assume伪指令却很是不懂。

比如已经定义了assume cs:code,ds:dat

但用debug观察的时候，发现ds段寄存器却没有相关联的数据。

必须在cs中写明： 

```assembly
MOV AX,Data
MOV DS,AXa
```

然后才能发现ds中有正确的数据。
于是疑惑，assume不是已经关联了ds嘛？

上网求助 = =、 然后找到答案。

编写程序，是写给编译软件的。由编译软件，编译成机器码，再去控制CPU。但是，编译软件，对assume语句，并不生成机器码。

所以，必须有mov ax,data，mov ds,ax，CPU才能受控。

**assume语句，是伪指令，仅仅是写给编译软件的。编译软件，并不把它生成机器码。**

**assume对除了CS以外的其它段寄存器，仅仅只是关联了段名，以便在访问段内变量时程序可以知道用哪个段寄存器，并没有在程序加载时将段地址装入段寄存器**。

**所以，将段地址装入段寄存器的工作，必须由用户在程序中自己编写代码，并在程序开始运行时执行代码完成装入工作。**
**仅仅对CS段寄存器，会在关联段名的同时，在程序加载时自动将段地址装入段寄存器。**



## 补充

前天知道了答案后，我以为assume ds:data 之类的指没有什么用，只是给程序员看的。

但今天发现不是这样的。 如果你在data中用了标号的话，则assume DS:Data不能省略。
比如：

```assembly
data segment
a db 1,2,3,4,5,6,7,8
b dw 0
data ends
```

a，b的后面没有“ ：”。

 如果你想在cs段中用数据标号访问数据，则必须在开头加上assume ds：data，否则会报错

Arror A2068:Can not address with segment register

不过就算在开头加上了assume ds：data，代码段中也不能少了mov ax,data，mov ds,ax。
**作用：用于标识默认段前缀**

**解释：assume 并不能改变ds等段寄存器的值，但他能改变编译器产生的汇编代码。比如：**

```assembly
assume ss：stack
stack segment
X: DB 0
stack ends
```

如果程序需要mov ax，[x],那么程序如何定位[x]呢？我们知道x只是一个偏移地址0，所以此时assume就相当于告诉编译器stack段的所有标号都与ss相关联，所以此时[x]就相当于ss:[0].

如果我们直接将这句改为mov ax,ss:[0]，那么前面不加assume也是可以的.这也是为什么[0]被编译器强制理解为立即数，而[标号]却被理解为标号里的内容的原因，因为**标号必须与段assume**，否则会报错cannot address with segment register.**而[0]无默认段，就只能被认为为立即数了.**

**所以，我们仍需在程序中将ss的值，用指令修改为stack，原因就是assume并不会修改段寄存器，这个由dos系统决定**，



**如果dos系统决定将段值编译进.exe文件头,并在加载进内存时根据文件头，修改段值，那么此时assume就相当于可以改变段值了.但是我调试的现实是ds、es指向psp头（psp详见16位exe程序加载过程），ss指向ds+0：00f0,cs指向ds+0:0100。**（这段没看懂）
————————————————
版权声明：本文为CSDN博主「手写的从前66」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_40627648/article/details/84063124












