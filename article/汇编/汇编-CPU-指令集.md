---
title: 汇编,CPU,指令集之间的关系[转]
comments: true
copyright: false
date: 2020-05-23 13:03:28
tags:
categories: 汇编
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0ly1gf2gks3n3dj30m80av756.jpg
toc: true
---



## 1.[指令集](https://marlous.github.io/2019/03/01/%E5%A4%84%E7%90%86%E5%99%A8%E3%80%81%E5%A4%84%E7%90%86%E5%99%A8%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%8C%87%E4%BB%A4%E9%9B%86%E5%85%B3%E7%B3%BB/)

1. 基本概念：

- 指令集是抽象概念、逻辑、设计规范，CPU 的软平台，一般公开。
- **CPU 执行计算任务时都需要遵从一定的规范，程序在被执行前都需要先翻译为 CPU 可以理解的语言。这种规范或语言就是指令集**（ISA，Instruction Set Architecture）。

   2.补充概念：

- CPU 架构是 CPU 厂商给属于同一系列的 CPU 产品定的一个规范，**主要目的是为了区分不同类型 CPU 的重要标示。**
- 目前市面上的 CPU 分类主要分有两大阵营，一个是 Intel、AMD 为首的复杂指令集 CPU，另一个是以 IBM、ARM 为首的精简指令集 CPU。
- 两个不同品牌的 CPU，其产品的架构也不相同，例如，Intel、AMD 的 CPU 是 X86 架构的，而 IBM 公司的 CPU 是 PowerPC 架构，ARM 公司是 ARM 架构。x86、ARM v8、MIPS 都是指令集的代号。

## 2 处理器架构（微架构）

1. 基本概念：

- 处理器架构（微架构）是具体实现、物理、设计方案的实现，CPU 的硬平台，一般保密。
- CPU 的基本组成单元即为核心（core）。多个核心可以同时执行多件计算任务，前提是这些任务没有先后顺序。核心的实现方式被称为微架构（microarchitecture）。

   2.补充概念：

- 如 Haswell、Cortex-A15 等都是微架构的称号。

## 3 指令集、处理器架构的关系

1. 基本概念（“架构” 的两层含义）：

- 软平台（CPU 的逻辑接口）：x86、ARM v8、MIPS … / 硬平台（直接体现在 CPU 的指标变化）：Ivy Bridge、Haswell、Cortex …
- 指令集是逻辑上的，处理器架构是物理上（实现）的；指令集可以用不同的架构实现，并不一定是决定关系。

## 4 指令集、汇编语言的关系

- **汇编语言是指令集的另一种表现形式（更易阅读）。**
- 软平台（CPU 的逻辑接口），**X86 架构、PowerPC 架构、ARM 架构等，其 X86、PowerPC、ARM 等也是指令集代号。**

<!--more-->

![undefined](http://ww1.sinaimg.cn/large/006eb5E0ly1gf2evzorkwj31zn311haa.jpg)



[源地址](https://marlous.github.io/2019/03/01/%E5%A4%84%E7%90%86%E5%99%A8%E3%80%81%E5%A4%84%E7%90%86%E5%99%A8%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%8C%87%E4%BB%A4%E9%9B%86%E5%85%B3%E7%B3%BB/)

## 5 补充 

1：计算机中的指令有微指令、机器指令和伪（宏）指令之分；
2：微指令是微程序级命令，属于硬件范畴；
3：伪指令是由若干机器指令组成的指令序列，属于软件范畴；
4：机器指令介于二者之间，处于硬件和软件的交界面；
5：令汇编指令是机器指令的汇编表示形式，即符号表示
机器指令和汇编指令一一对应，它们都与具体机器结构有关，都属于机器级指令

[参考](https://blog.csdn.net/qq_44116998/article/details/102829047)