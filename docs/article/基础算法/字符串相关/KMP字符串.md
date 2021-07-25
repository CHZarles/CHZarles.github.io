---
title: KMP字符串(例题)
tags:
  - KMP
  - 题解
categories:
  - 基础算法
  - 数据结构
comments: true
copyright: false
date: 2020-01-12 19:56:25
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [KMP字符串](   https://www.acwing.com/problem/content/833/  )

给定一个模式串S，以及一个模板串P，所有字符串中只包含大小写英文字母以及阿拉伯数字。

模板串P在模式串S中多次作为子串出现。

求出模板串P在模式串S中所有出现的位置的起始下标。


#### 输入格式

第一行输入整数N，表示字符串P的长度。
<!--more-->

第二行输入字符串P。

第三行输入整数M，表示字符串S的长度。

第四行输入字符串S。

#### 输出格式

共一行，输出所有出现位置的起始下标（下标从0开始计数），整数之间用空格隔开。

#### 数据范围

1≤N≤1e4
1≤M≤1e5

#### 输入样例

```
3
aba
5
ababa
```

#### 输出样例

```
0 2
```


 </br>

## 算法/复杂度

复杂度 ：O(m+n)

详细算法：**{% post_link 基础算法/数据结构/KMP算法 %}** 

## C++ 代码


```c++
#include<iostream>
#include<cstring>
using namespace std;
const int N = 10010;
int ne[N];

void get_next(string pattern,int l)
{
    int j = -1;
    ne[0] = -1;
    
    //维护j一直指向上一个ne[i]
    //维护i一直指向新增字符的指针i
    
    /*
    ne[] 数组是 子串 pattern[0~k] 最长相等前后缀的前缀的最后一位的下标。
    因为最长相等前后缀不能是子串本身，所以i从1开始.
    */
    for(int i = 1 ; i < l ; i++ )
    {
        //当j==-1或找到s[i] == s[j+1]时，停止循环
        while(j!=-1&&pattern[i]!=pattern[j+1]) j = ne[j];
        if(pattern[i]==pattern[j+1]) j++;
        ne[i] = j;
    }
}

void get_match(string s,string p,int pl,int sl)
{
    //维护j一直指向"匹配串"上次被匹配到的位置
    int j = -1;
    //维护i一直指向文本串当前正在匹配的字符
    for(int i = 0 ; i < sl ; i++ )
    {
        while(j!=-1&&s[i]!=p[j+1]) j = ne[j];
        
        if(s[i]==p[j+1]) j++;
        
        //判断j是不是指向匹配串最后一个下标
        if(j==pl-1) 
        {
            cout<<i-pl+1<<" ";
            j = ne[j];
        }
    }
}

int main()
{
    string s,p;
    int pl,sl;
    cin>>pl>>p>>sl>>s;
    get_next(p,pl);
    get_match(s,p,pl,sl);
    return 0;
}
```


