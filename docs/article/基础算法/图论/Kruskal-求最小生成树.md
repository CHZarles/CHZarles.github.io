---
title: Kruskal 求最小生成树
tags:
  - 题解
categories:
  - 基础算法
  - 图论
comments: true
copyright: false
date: 2019-12-23 17:56:31
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：Kruskal算法求最小生成树]( https://www.acwing.com/problem/content/861/  )

给定一个n个点m条边的无向图，图中可能存在重边和自环，边权可能为负数。

求最小生成树的树边权重之和，如果最小生成树不存在则输出impossible。

给定一张边带权的无向图G=(V, E)，其中V表示图中点的集合，E表示图中边的集合，n=|V|，m=|E|。

由V中的全部n个顶点和E中n-1条边构成的无向连通子图被称为G的一棵生成树，其中边的权值之和最小的生成树被称为无向图G的最小生成树。


#### 输入格式

<!--more-->
第一行包含两个整数n和m。

接下来m行，每行包含三个整数u，v，w，表示点u和点v之间存在一条权值为w的边。

#### 输出格式

共一行，若存在最小生成树，则输出一个整数，表示最小生成树的树边权重之和，如果最小生成树不存在则输出impossible。

#### 数据范围

1≤n≤1e5,
1≤m≤2∗1e5,
图中涉及边的边权的绝对值均不超过1000。

#### 输入样例

```
4 5
1 2 1
1 3 2
1 4 3
2 3 2
3 4 4
```

#### 输出样例

```
6
```




## 算法/复杂度

**算法流程 :**

**用结构体存边，结构体记录了边的点的所在集合，边的权值，把边看成图的基本结构。**

**把边从小到达排序，依次选择边，如果两个端点不在同一个集合（避免成环），那选择合法，可以选择，然后将端点a合并到端点b。（涉及并查集操作），如果两个端点在同一个集合，跳过这条边。**

**重复上面步骤，直到所有边都遍历完。**



**复杂度:mlog(N)(排序的复杂度)**

## C++ 代码


```c++
#include<iostream>
#include<algorithm>
using namespace std;
const int N = 100010 , M = 2*100010, INF = 0x3f3f3f3f ;
int n,m;
//并查集
int p[N];
//存边
struct Edge
{
    int a,b,w;
    bool operator < (Edge& Z)const
    {
        return w<Z.w;
    }
}edges[M];

//查找操作
int find(int x)
{
    if(p[x]!=x) p[x] = find(p[x]);
    return p[x];
}
//初始化并查集
void init()
{
    for(int i = 0 ; i < N ;i ++)
        p[i] = i;
}

//合并操作
void merge(int a,int b)
{
    p[find(a)] = p[b];
}

int kruskal()
{
    sort(edges,edges+m);
    init();
    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;

        a = find(a), b = find(b);
        if (a != b)
        {
            p[a] = b;
            res += w;
            cnt ++ ;
        }
    }

    if (cnt < n - 1) return INF;
    return res;


}


int main()
{
    cin>>n>>m;
    for(int i = 0; i <m ; i++)
    {
        cin>>edges[i].a>>edges[i].b>>edges[i].w;
    }

    int k = kruskal();
    if(k == INF)
        puts("impossible");
    else
        cout<<k<<endl;
    return 0;
}

```


