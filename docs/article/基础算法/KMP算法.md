---
title: KMP算法
comments: true
copyright: false
date: 2020-01-12 10:59:26
tags:
	- 基础算法
	- 字符串
	- KMP
categories:
	- 基础算法
	- 数据结构
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---





### 记忆提要

**概念：**

- 要在一个文本文件里查找关键词"wind" ,  那么"wind"叫 "匹配串" , 用pattern表示。假设，pattern的长度为L。
- next数组: next[k]记录了pattern子串pattern[0~k] (k<=L) , 的最长公共前后缀的前缀的最后一位字母的下标。
<!--more-->
- 最长公共前后缀不能是pattern子串本身,但可以重叠。



### 求解next数组

假设有一个pattern是 : ababaab

pattern的各子串和相关的next的数组的值如下。

如果某个子串pattern[0~k]不存在公共前后缀，那么next[k] = -1 。

显然 next[0] = -1;

| 子串pattern[0~k] | 前缀        | 后缀        | next[k] |
| ---------------- | ----------- | ----------- | ------- |
| a                | NuLL        | NuLL        | -1      |
| ab               | NuLL        | NuLL        | -1      |
| aba              | **a**ba     | ab**a**     | 0       |
| abab             | **ab**ab    | ab**ab**    | 1       |
| ababa            | **aba**ba   | ab**aba**   | 2       |
| ababaa           | **a**babaa  | ababa**a**  | 0       |
| ababaab          | **ab**abaab | ababa**ab** | 1       |



**流程：**

用递推的方法求解next数组是最快的。



**步骤**

假设子串 pattern[0~k] 的next[k] 已知（k>=0）

求子串pattern[0~k+1]的next[k+1]。用 指针j 指向 next[k] 所存储的下标(即 j = next[k]) ，用指针i指向下标k+1.

 比较 pattern[j+1] 和 pattern[i] 。

- 如果pattern[j+1] == pattern[i]   

  -  则 next[k+1] =next[k]+1 ;

  ![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1gatxssurjtj30qy0cmaac.jpg)

- 如果pattern[j+1] ！= pattern[i] ：

  - 让 j 回退：while( j != -1 && pattern[j+1] 和==pattern[i] )  j = next[j]  
  - 如果pattern[j+1] == pattern[i]    则 next[k+1] = next[k]+1 ；
  - 如果pattern[j+1] ！= pattern[i]  则 next[k+1] = -1;

  ![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1gatxts3efsj312b0edaao.jpg)



### 匹配文本

文本匹配还是有两个指针，看图。

![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1gatyw92t4mj30s706kjre.jpg)

**指针j  指向当前匹配串已经匹配好的前缀的最后一个字符**，**指针i  指向当前文本串正在匹配的字符**。

- 1.如果pattern[j+1] == text[i]  指针i 和 指针j 都向前移动一位，继续向前移动 匹配。
  - 1.1如果j移动后指向pattern的最后一个字符，则说明在文本中找到一个pattern串。
  - 1.2如果在文本中想继续向前寻找pattern，就要先让 **指针j回退** ( 回退策略同计算next数组) , 然后回到步骤1
- 2.如果满足 pattern[j+1] != text[i]  , 指针j 回退，回退策略和前面求next的回退策略一样。
  - 2.1如果j回退到-1后，还是不满足 pattern[j+1] != text[i]，指针i 向前移动一位，然后回到步骤1。



### 一个完整的例子

原博客地址_REference：https://www.cnblogs.com/SYCstudio/p/7194315.html

首先我们还是从0开始匹配：
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP2-2.gif)
此时，我们发现，A的第5位和B的第5位不匹配（注意从０开始编号)，此时i=5,j=5，那么我们看next[j-1]的值：

> next[5-1]=2;

这说明我们接下来的匹配只要从B串第２位开始（也就是第３个字符）匹配，因为前两位已经是匹配的啦，具体请看图：
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP3.gif)
然后再接着匹配：
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP4.gif)
我们又发现，A串的第13位和B串的第10位不匹配，此时i=13,j=10，那么我们看next[j-1]的值：

> next[10-1]=4

这说明B串的0~3位是与当前(i-4)~(i-1)是匹配的，我们就不需要重新再匹配这部分了，把B串向后移，从Ｂ串的第４位开始匹配：
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP5.gif)

这时我们发现A串的第13位和B串的第4位依然不匹配
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_%E5%9B%BE%E7%89%8728.png)
此时i=13,j=4，那么我们看next[j-1]的值：

> next[4-1]=1

这说明B串的第0位是与当前i-1位匹配的，所以我们直接从B串的第1位继续匹配：
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP6.gif)
但此时B串的第1位与A串的第13位依然不匹配
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_%E5%9B%BE%E7%89%8733.png)
此时，i=13,j=1,所以我们看一看next[j-1]的值:

> next[1-1]=0

好吧，这说明已经没有相同的前后缀了，直接把B串向后移一位，直到发现B串的第0位与A串的第i位可以匹配（在这个例子中，i=13）
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP7.gif)
再重复上面的匹配过程，我们发现，匹配成功了！
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP8.gif)

这就是KMP算法的过程。
另外强调一点，当我们将B串向后移的过程其实就是i++,而当我们不动B，而是匹配的时候，就是i++,j++，这在后面的代码中会出现，这里先做一个说明。

最后来一个完整版的（话说做这些图做了好久啊！！！！）：
![此处输入图片的描述](https://gitee.com/chzarles/images/raw/master/imgs/o_KMP9.gif)



------

 