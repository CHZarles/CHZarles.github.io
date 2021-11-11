---
title: 线性DP-数字三角形
comments: true
copyright: false
date: 2019-10-23 17:06:14
tags: [基础算法,动态规划]
categories: [基础算法,动态规划]
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述： 数字三角形 ](  https://www.acwing.com/problem/content/900/   )

给定一个如下图所示的数字三角形，从顶部出发，在每一结点可以选择移动至其左下方的结点或移动至其右下方的结点，一直走到底层，要求找出一条路径，使路径上的数字的和最大。

```c++
        7
      3   8
    8   1   0
  2   7   4   4
4   5   2   6   5
```


#### 输入格式

第一行包含整数n，表示数字三角形的层数。
<!--more-->

接下来n行，每行包含若干整数，其中第 i 行表示数字三角形第 i 层包含的整数。

#### 输出格式

 输出一个整数，表示最大的路径数字和。 

#### 数据范围

 1≤n≤500
−10000≤三角形中的整数≤10000

#### 输入样例

```c++
5
7
3 8
8 1 0 
2 7 4 4
4 5 2 6 5
```

#### 输出样例

```c++
30
```


## 基本思考框架

**要用集合的思想来思考DP，首先要思考怎么用集合来表示这个问题的各个状态（怎么用数组表示这个问题的各个状态）。**以样例为例子，其实我们只要给数字三角形的各个位置定位即可。接下来，我们再用集合的思考框架来思考。

![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1g88fu4gnqqj30ay06zt8n.jpg)

**思考框架**

![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1g88gbgqpuoj30xu0jxq4l.jpg)



## C++ 代码


```c++
#include<iostream>
using namespace std;
const int N = 510;
//注意有负权
const int inf = -1e9;
int f[N][N];
int a[N][N];
int n;

int main()
{
    cin>>n;
    //输入三角形
    for(int i = 1 ; i <= n ;i ++)
        for(int j = 1 ; j <= i ;j++)
            cin>>a[i][j];
    
    /*
    初始化集合f[i][j]，因为有负数,注意边界f[i][0]和f[i][n+1]要初始成-inf，
    不能不管  ，不然节点是权值是负的时候的时候会wa
    */
    
    for(int i = 0 ; i <= n ; i++)
        for(int j = 0 ; j <= i+1 ;j++)
            f[i][j] = inf;
            
    f[1][1] = a[1][1];
    for(int i = 2 ; i <= n ;i++)
        for(int j = 1 ; j <= i ;j++)
             f[i][j] = max(f[i-1][j-1]+a[i][j] , f[i-1][j]+a[i][j]);
    
    int res = inf;
    for(int i = 1 ; i <= n ; i ++ )
        res =max(res , f[n][i]);
        
    cout<<res<<endl;
    return 0;
}

```



------



