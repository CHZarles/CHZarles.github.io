---
title: 字符串hash
tags:
  - 字符串hash
categories:
  - 基础算法
  - 数据结构
comments: true
copyright: false
date: 2020-01-14 14:31:44
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：字符串哈希](   https://www.acwing.com/problem/content/843/ )

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gaw329b0uvj30vg04274u.jpg)


#### 输入格式

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gaw32py0b1j30pk05ygme.jpg)

#### 输出格式

对于每个询问输出一个结果，如果两个字符串子串完全相同则输出“Yes”，否则输出“No”。

<!--more-->
每个结果占一行。

#### 数据范围

1≤n,m≤1e5

#### 输入样例

```
8 3
aabbaabb
1 3 5 7
1 3 6 8
1 2 1 2
```

#### 输出样例

```
Yes
No
Yes
```




## 算法/复杂度

核心思想：将字符串看成P进制数，P的经验值是131或13331，取这两个值的冲突概率低。字母不能映射成0。

小技巧：取模的数用2^64，这样直接用unsigned long long存储，溢出的结果就是取模的结果



```cc
//模板 ， base = 131 或 13331 或 ......
typedef unsigned long long ULL;
ULL h[N], p[N]; // h[k]存储字符串前k个字母的哈希值, p[k]存储 P^k mod 2^64

// 初始化
p[0] = 1;
for (int i = 1; i <= n; i ++ )
{
    h[i] = h[i - 1] * base + str[i];
    p[i] = p[i - 1] * base;
}

// 计算子串 str[l ~ r] 的哈希值
ULL get(int l, int r)
{
    return h[r] - h[l - 1] * p[r - l + 1];//是r-l,看清楚
}

```



## C++ 代码


```c
#include<iostream>
using namespace std;
const int N = 1000010;
unsigned long long f[N],h[N];
int main()
{
    int n,m;
    string s;
    cin>>n>>m>>s;
    
    h[0]=1;
    for(int i=n;i>0;i--)
        s[i]=s[i-1];
        
    for(int i=1;i<=n;i++)
    {
        f[i]=f[i-1]*131+s[i]%131;
        h[i]=h[i-1]*131;
    }
    while(m--)
    {
        int l1,l2,r1,r2;
        cin>>l1>>r1>>l2>>r2;
        if(f[r1]-f[l1-1]*h[r1-l1+1]==f[r2]-f[l2-1]*h[r2-l2+1])
            puts("Yes");
        else
            puts("No");
    }
}
```
