---
title: ' Floyd求最短路'
tags:
  - 题解
categories:
  - 基础算法
  - 图论
comments: true
copyright: false
date: 2019-12-10 15:49:43
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：Floyd求最短路](  https://www.acwing.com/problem/content/856/  )

给定一个n个点m条边的有向图，图中可能存在重边和自环，边权可能为负数。

再给定k个询问，每个询问包含两个整数x和y，表示查询从点x到点y的最短距离，如果路径不存在，则输出“impossible”。

数据保证图中不存在负权回路。


#### 输入格式

第一行包含三个整数n，m，k

<!--more-->
接下来m行，每行包含三个整数x，y，z，表示存在一条从点x到点y的有向边，边长为z。

接下来k行，每行包含两个整数x，y，表示询问点x到点y的最短距离。

#### 输出格式

共k行，每行输出一个整数，表示询问的结果，若询问两点间不存在路径，则输出“impossible”。

#### 数据范围

1≤n≤200,
1≤k≤n2
1≤m≤20000,

图中涉及边长绝对值均不超过10000。

#### 输入样例

```
3 3 2
1 2 1
2 3 2
1 3 1
2 1
1 3
```

#### 输出样例

```
impossible
1
```




## 算法/复杂度

**注意松弛操作那里的循环顺序**

## C++ 代码


```c++
#include<iostream>
#include<algorithm>
using namespace std;
const int N = 210,inf=0x3f3f3f3f;
int d[N][N];
int n,m,k;

void floyd()
{
    //暴力枚举，这里一定要是先t后i后j
    for(int t = 1;t<=n;t++)
    for(int i =1;i<=n;i++)
    for(int j =1;j<=n;j++)
        d[i][j]=min(d[i][j],d[i][t]+d[t][j]);
    
}

int main()
{
    cin>>n>>m>>k;
    
    //不能那么粗暴地初始化
    //memset(d,inf,sizeof d);
    
    for(int i = 1 ; i<=n ;i++)
    for(int j =1;j<=n ;j++)
        if(i == j) 
            d[i][j]=0;
        else 
            d[i][j]=inf;
        
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        d[a][b] = min(d[a][b],c);
    }
    
    floyd();
    
    while(k--)
    {
        int a,b;
        cin>>a>>b;
        if(d[a][b]>inf/2)
            puts("impossible");
        else
            cout<<d[a][b]<<endl;
    }
    return 0;
}
```
