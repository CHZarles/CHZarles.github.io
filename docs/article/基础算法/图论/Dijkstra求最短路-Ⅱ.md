---
title: Dijkstra求最短路 Ⅱ
tags:
  - 图论
categories:
  - 基础算法
  - 图论
comments: true
copyright: false
date: 2019-12-07 20:23:50
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：Dijkstra求最短路 Ⅱ](   https://www.acwing.com/problem/content/852/)

给定一个n个点m条边的有向图，图中可能存在重边和自环，所有边权均为非负值。

请你求出1号点到n号点的最短距离，如果无法从1号点走到n号点，则输出-1。


#### 输入格式

第一行包含整数n和m。

接下来m行每行包含三个整数x，y，z，表示存在一条从点x到点y的有向边，边长为z。

<!--more-->
#### 输出格式

输出一个整数，表示1号点到n号点的最短距离。

如果路径不存在，则输出-1。

#### 数据范围

1<n,m≤1e5,
图中涉及边长均不小于0，且不超过10000。

时/空限制：1s / 64MB

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

**1.这是堆优化版的dijkstra**

**2.这题最大有1e5个点，用邻接矩阵存图必爆内存(1e5乘1e5乘4 >64乘1024乘1024乘8)。所以只能用这种邻接表存图的写法。(很明显，比起朴素版的dijikstra这种写法更通用)**

## C++ 代码


```c++
#include<iostream>
#include<vector>
#include<cstring>
#include<queue>
using namespace std;
typedef pair<int,int> pii;
const int N = 100010;
const int inf = 0x3f3f3f3f;
int h[N],w[N],e[N],ne[N],idx;
int dist[N],m,n;
bool st[N];

void add(int a,int b,int c)
{
    e[idx] = b,w[idx] = c,ne[idx] = h[a],h[a] = idx++;
}


int dj()
{
    priority_queue<pii,vector<pii>,greater<pii>> heap;
    heap.push({0,1});
    dist[1] = 0;
    
    while(heap.size())
    {
        //取出队中与源点最近的点
        auto p = heap.top();
        heap.pop();
        
        int d = p.first,ver = p.second;
        if(st[ver])
            continue;
            
        //纳入麾下    
        st[ver] = true;
        
        //更新视野+优化内部
        for(int i = h[ver];i!=-1;i = ne[i])
        {
            int j = e[i];
            if(dist[j]>d+w[i])
            {
                dist[j] = d + w[i];
                heap.push({dist[j],j});
            }
            
        }
    }
    
    if(dist[n]==inf)
        return -1;
    else
        return dist[n];
}


int main()
{
    memset(dist,inf,sizeof dist);
    //别忘了
    memset(h,-1,sizeof h);
    cin>>n>>m;
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        //这里没有处理重边的问题
        add(a,b,c);
    }
    cout<<dj();
}
```


