---
title: 调用Dos接口实现输入输出
comments: true
copyright: false
date: 2020-05-24 19:48:22
tags:
categories: 汇编
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0ly1gf3tyrux0uj306j08fmxb.jpg
toc: true
---





## 介绍

在汇编语言中，凡是涉及到键盘输入、屏幕显示等输入输出操作，**都可以用软件中断指令 `INT n` 的功能调用来实现**。所谓功能调用是计算机系统设计的简单 I / O 子程序，能方便地访问系统的硬件资源。

在微机系统中，**功能调用分两个层次，最底层的是 BIOS 功能调用，其次是 DOS 功能调用，它们都是通过软件中断指令 `INT n` 来进行调用的。除了用 INT 指令实现输入和显示之外，还可以通过直接写显示缓冲区的方式显示字符。**

#### （1）DOS 功能调用

DOS 的功能调用采用 `INT 21H` 指令，调用时要求在 **AH** 中提供功能号，**在指定的寄存器和存储单元中提供调用必需的参数和缓冲区地址，执行后系统在 AL 中放入返回参数**。

常用的 DOS 功能调用有 5 个：

- 1 号 DOS 功能调用：键盘输入 1 个字符
- 2 号 DOS 功能调用：显示器输出 1 个字符
- 9 号 DOS 功能调用：显示字符串
- 10 号 DOS 功能调用：键盘输入缓冲区
- 4CH 号 DOS 功能调用：返回 DOS 控制

**`注意`**：I/O 处理操作的都是 ASCⅡ 码，对于键盘输入的数字，做计算时需将 ASCⅡ 码转变为二进制数，输出显示数据时需将二进制数转为 ASCⅡ 码。数字 0～9 的 ASCII 码为 30H～39H，可以看出两者之间相差 30H。

<!--more-->



## 例子

**显示两行字符串（回车换行功能）。第 1 行为“Input x:”，第 2 行为“Output y=x+1:”。从键盘输入 x，输出 y = x + 1 的值。**

我发现我电脑的环境跑不了这个程序(秃头.jpg)

```assembly
data segment
    mess1 db 'Input x:','$' ; $是表示字符串结束的符号，如果要输出的话必须有这个符号，否则输出将不会停止。(百度说的)
    mess2 db 0ah,0dh, 'Output y:$' ;0ah和0dh是换行和回车
    y db ?
data ends

code segment
    assume cs:code,ds:data

    start:
    mov ax,data ;初始化数据段段地址
    mov ds,ax

    mov dx,offset mess1; mess1表示地址，取偏移量放到dx中，作为下面中断服务的参数
    mov ah,9 ; 将中断服务的功能号存入ah,0号表示输出字符串
    int 21h  ; 调用服务，显示提示信息'Input x:'

    mov ah,1 ; 将中断服务的功能号存入ah,1号表示键入一个字符
    int 21h ; 调用服务，键盘输入，键入的值在al

    add al,1
    mov y,al ; 这里y做了地址

    mov dx,offset mess2 ;同上
    mov ah,9 
    int 21h

    mov dl,y     ;显示的字符要放入dl,显示x+1的值
    mov ah,2     ;将中断服务的功能号存入ah,2号表示显示一个字符
    int 21h

    mov ah,4ch
    int 21h

code ends
end start

```








