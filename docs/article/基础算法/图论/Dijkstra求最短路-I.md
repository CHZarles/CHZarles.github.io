---
title: Dijkstra求最短路 I
tags:
  - 基础算法
categories:
  - 基础算法
  - 图论
comments: true
copyright: false
date: 2019-12-07 20:00:51
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：Dijkstra求最短路 I](https://www.acwing.com/problem/content/851/  )

给定一个n个点m条边的有向图，图中可能存在重边和自环，所有边权均为正值。

请你求出1号点到n号点的最短距离，如果无法从1号点走到n号点，则输出-1。


#### 输入格式

第一行包含整数n和m。

接下来m行每行包含三个整数x，y，z，表示存在一条从点x到点y的有向边，边长为z。

<!--more-->
#### 输出格式

输出一个整数，表示1号点到n号点的最短距离。

如果路径不存在，则输出-1。

#### 数据范围

1<= n <=500
1<= m <=1e5,
图中涉及边长均不超过10000。

#### 输入样例

```
3 3
1 2 2
2 3 1
1 3 4
```

#### 输出样例

```
3
```




## 算法/复杂度

**1.这题是朴素版的Dijkstra写法的模板题，这个写法是用邻接矩阵存图的，适合点比较少的情况**

**2.，Dijkstra只能处理正权图**

## C++ 代码


```c++
#include<iostream>
#include<cstring>
using namespace std;
const int N = 600 , inf = 0x3f3f3f3f;
int g[N][N];
//dist[x]=a,表示点x到源点的最短距离是a
int dist[N];
bool st[N];
int n,m;

int dj()
{
    //1.初始化距离
    memset(dist,inf,sizeof dist);
    //注意不要加st[1] = true
    dist[1]=0;

    //2.将点纳入集合st[i]
    //坑,这里循环n-1次就行
    for(int i=0 ; i<n-1 ; i++)
    {
        int t = -1;//t记录"最佳新人"

        //2.1筛选不在集合st的距离源点最近的点。
        for(int j=1 ; j<=n ; j++)
            if(!st[j]&&(t==-1||dist[j]<dist[t]))
                t = j;

        //2.2纳入集合
        st[t] = true;

        //2.3更新源点到各点的最短距离
        for(int j=1 ; j<=n ; j++)
            dist[j] = min(dist[j],dist[t]+g[t][j]);
    }

    //3.检查终点有没有纳入st
    if(dist[n]==inf) return -1;
        else
    return dist[n];
}

int main()
{
    memset(g,inf,sizeof g);
    cin>>n>>m;
    while(m--)
    {
        int a,b,w;
        cin>>a>>b>>w;
        g[a][b]=min(g[a][b],w);
    }

    cout<<dj()<<endl;
    return 0;
}

```


