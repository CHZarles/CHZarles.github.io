---
title: ' Prim算法求最小生成树'
comments: true
copyright: false
date: 2019-12-11 21:30:26
tags:
	- 题解
categories:
	- 基础算法
	- 图论
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述： Prim算法求最小生成树]( https://www.acwing.com/problem/content/860/  )

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

1≤n≤500,
1≤m≤1e5,
图中涉及边的边权的绝对值均不超过10000。

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

**Tip: s[X]标记点X是否在集合S里 , dist[X]记录某个点到集合S的最短距离 ,res记录最小生成树权值**

**这题点很少,邻接矩阵存图,一般求最小生成树都是用朴素版prim**

```tcl
算法路程：
1.预处理邻接矩阵-->初始化邻接矩阵
2.prim:
	预处理dist数组成inf，dist[1] = 0;
	for 循环 n 次(最多收纳n个点到s)
		选出不在s中的离s最近的点t
		如果 dist[t] == inf 
			说明图不联通 
			退出算法
		更新 res += dist[t]
		更新 s[t] = True
		for  k 循环 n 次
			用 g [t] [ k] 更新每个点到集合s的最短距离
```

**复杂度：O(n^2)**



## C++ 代码


```c++
//图中可能存在重边和自环，边权可能为负数
#include<iostream>
#include<cstring>
using namespace std;
const int N = 510 ,inf = 0x3f3f3f3f;
int g[N][N],dist[N];
bool s[N];
int n,m,res;

void prim()
{
    memset(dist,inf,sizeof dist);
    //这里初始化一下dist[0]
    dist[1] = 0;
    
    for(int i = 0 ; i < n ; i ++)
    {
        int t = -1;
        
        
        for(int j = 1 ; j <= n ; j++)
            if(!s[j]&&(t==-1||dist[j]<dist[t]))
                t = j;
                
        if(dist[t]==inf)
        {
            res = -1;
            return ;    
        }
        
        s[t] = true;
        res += dist[t];
        
        for(int k = 1; k <= n  ; k ++ )
            dist[k] = min(dist[k],g[k][t]);
    }
}

int main()
{
  
    cin>>n>>m;
    
    for(int a = 1 ; a <= n  ; a++ )
    for(int b = 1 ; b <= n  ; b++ )
        {   
            g[a][b] = g[b][a] = inf;
            if(a==b) g[a][b] = g[b][a] = 0;
        }
      
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        g[a][b] = min(g[a][b],c);
        g[b][a] = g[a][b];
    }
    prim();
    if(res == -1)
        puts("impossible");
    else
        cout<<res<<endl;
    return 0;
}

```



### 朴素prim的写法和朴素版Dijkstra的写法比较

**{%post_link  基础算法/图论/Dijkstra求最短路-I%}**

**两个算法写法结构很相似，我们要重点注意的是，两个dist[x]的意义是不同的。**





------


