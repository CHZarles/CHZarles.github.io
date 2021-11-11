---
title: bellman-ford_ 有边数限制的最短路
comments: true
copyright: false
date: 2019-12-09 16:57:54
tags: 
	- 图论
	- 基础算法
categories:
	- 基础算法
	- 图论
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：有边数限制的最短路](   https://www.acwing.com/problem/content/855/  )

给定一个n个点m条边的有向图，图中可能存在重边和自环， **边权可能为负数**。

请你求出从1号点到n号点的**最多经过k条边**的最短距离，如果无法从1号点走到n号点，输出impossible。

注意：图中可能 **存在负权回路** 。


#### 输入格式

第一行包含三个整数n，m，k。
<!--more-->

接下来m行，每行包含三个整数x，y，z，表示存在一条从点x到点y的有向边，边长为z。

#### 输出格式

输出一个整数，表示从1号点到n号点的最多经过k条边的最短距离。

如果不存在满足条件的路径，则输出“impossible”。

#### 数据范围

1≤n,k≤500,
1≤m≤10000,
任意边长的绝对值不超过10000。

#### 输入样例

```
3 3 1
1 2 1
2 3 1
1 3 3
```

#### 输出样例

```
3
```




## 算法/复杂度

**有几个要注意的点（先看代码再回来理解）：**

1.  **bellman-ford是可以处理有负权图的，因为限定了K, 所以不用担心有负环而造成的死循环。**
2. **因为有负环，k很大时，有可能会求出某个 dist[i] 是 “ -∞ ” (负无穷) 。** 
3. **下面代码的里层循环记得用backup备份,并用于更新操作。**
4. **有负权回路的图，1->n的最短路"不一定"存在，可以改写bellford-man来判断图中1->n的路程有无负权回 路 但实际一般用spfa的写法来判断。**
5. **这个写法，对存图没什么要求**

## C++ 代码


```c++
#include<iostream>
#include<cstring>
using namespace std;
const int N = 550 , M = 100010 ,inf = 0x3f3f3f3f ;
int k,n,m,dist[N];

//边a->b,value == c
struct node
{
    int a,b,c;
}edges[M];

int bfm()
{
    //step 1 *初始化dist[1]
    int backup[N];
    memset(dist,inf,sizeof dist);
    dist[1] =  0;
    
    //step 2 *k次循环
    //这个循环可以理解为：发现小路"backp[a]",走这条路，看能不能从起点到点b的路程更近。
    for(int i = 0 ; i < k ;i ++)
    {
        memcpy(backup,dist,sizeof dist);
        for(int j = 0; j < m ; j++)
        {   
            //记得用backup数组
            dist[edges[j].b] = min( dist[edges[j].b],backup[edges[j].a]+edges[j].c);
            //printf("dist[%d] = %d\n" , edges[j].b,  dist[edges[j].b]);
        }
    }
    
    //step 3 *判断是dist[n]>inf/2
    if(dist[n]>inf/2)
        return -1;
    else
        return dist[n];
}

int main()
{
    cin>>n>>m>>k;
    for(int i = 0 ; i < m ;i ++)
    {
        cin>>edges[i].a>>edges[i].b>>edges[i].c;
    }
    
    if(bfm()==-1) puts("impossible");
    else cout<<bfm();
    return 0;
}
```

------


