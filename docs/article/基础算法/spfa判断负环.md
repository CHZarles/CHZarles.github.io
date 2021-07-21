---
title: spfa判断负环
tags:
  - 图论
categories:
  - 基础算法
  - 图论
comments: true
copyright: false
date: 2019-12-10 15:07:27
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：spfa判断负环](  https://www.acwing.com/problem/content/854/  )

给定一个n个点m条边的有向图，图中可能存在**重边和自环**， **边权可能为负数**。

请你判断图中是否存在负权回路。


#### 输入格式

第一行包含整数n和m。

接下来m行每行包含三个整数x，y，z，表示存在一条从点x到点y的有向边，边长为z。

<!--more-->
#### 输出格式

如果图中**存在**负权回路，则输出“Yes”，否则输出“No”。

#### 数据范围

1≤n≤2000,
1≤m≤10000
图中涉及边长绝对值均不超过10000。

#### 输入样例

```
3 3
1 2 -1
2 3 4
3 1 -4
```

#### 输出样例

```
Yes
```




## 算法/复杂度

```tcl
算法流程：
	1.将所有顶点放入队列("确保多起点，有环也会被更新到")，所有点的st[i]置true
	2.队列循环：（非空）
		节点出队
		st[i]置false
		遍历点i连着的点j
		如果更换路径令距离dist[j]变短了
		更新dist[j]
		cnt[j] = cnt[i] + 1
		如果cnt[j]>=n  有负环 return true
		如果j当前不在队里
			j入队
			更新st[j] = true
	3.队列空了，说明没有负环，return false			
														
```



**Tip:**

1. **cnt[x]表示从1号点到达该点x的最优路径有几条边。**

2. **dist[x]表示当前1号点到x点的最短长度**

3. **如果存在负环,那么任意两个点之间的距离必然是"负无穷"(因为可以在负环一直绕一直绕,让权值一直减少，进一步可知，dist[x]，和cnt[x]会一直被更新)**

   

**复杂度 :O(nm)**

## C++ 代码


```c++
/*推荐写法 ：STL的队列 运行时间： 2226 ms*/
#include<iostream>
#include<cstring>
#include<queue>
using namespace std;
const  int N = 2010 , M = 10010 ;
int st[N],cnt[M],dist[N];
int h[N],e[M],ne[M],w[M],idx;
int n,m;

void add(int a,int b,int c)
{
    e[idx] = b,ne[idx] = h[a],w[idx] = c, h[a] = idx++;
}

bool spfa()
{
    //初始化dist为inf不是必要的
    memset(dist,0x3f,sizeof dist);
    //循环队列
    
    queue<int> q;
    //多个起
    for(int i=1;i<=n;i++)
    {
        st[i] = true;
        q.push(i);
    }
    
    while(q.size())
    {
        int p = q.front();
        q.pop();
        st[p] = false;
        for(int i = h[p];i!=-1;i=ne[i])
        {
            int j = e[i];
            if(dist[j]>dist[p]+w[i])
            {
                dist[j] = dist[p]+w[i];
                cnt[j] = cnt[p] + 1;
                //有n条边，是环无疑
                if(cnt[j]>=n) return true;
                if(!st[j])
                {
                    st[j] = true;
                    q.push(j);
                }
            }
        }
    }
    return false;
    
}

int main()
{
    cin>>n>>m;
    memset(h,-1,sizeof h);
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    if(spfa())
        puts("Yes");
    else
        puts("No");
        
    return 0;
}
```



```c++
/*数组模拟循环队列,运行时间： 1214 ms*/
#include<iostream>
#include<cstring>
#include<queue>
using namespace std;
const  int N = 2010 , M = 10010 ;
bool st[N];
int cnt[M],dist[N];
int h[N],e[M],ne[M],w[M],idx;
int n,m;

void add(int a,int b,int c)
{
    e[idx] = b,ne[idx] = h[a],w[idx] = c, h[a] = idx++;
}

bool spfa()
{
    //初始化dist为inf不是必要的
    memset(dist,0x3f,sizeof dist);
    //循环队列
    int q[N],hh = 0,tt = 0;
    
    for(int i=1;i<=n;i++)
    {
        q[tt++] = i;
        st[i] = true;
    }
      
    while(hh!=tt)
    {
        //特殊情况，不得不将这句放进来
        if(tt==n)
        tt = 0;
        
        int p = q[hh++];
        if(hh==n)
            hh = 0;
        st[p] = false;
        for(int i = h[p];i!=-1;i=ne[i])
        {
            int j = e[i];
            if(dist[j]>dist[p]+w[i])
            {
                dist[j] = dist[p]+w[i];
                cnt[j] = cnt[p] + 1;
                //有n条边，是环无疑
                if(cnt[j]>=n) return true;
                if(!st[j])
                {
                    st[j] = true;
                    q[tt++] = j;
                    if(tt==n)
                        tt = 0;
                }
            }
        }
    }
    return false;
    
}

int main()
{
    cin>>n>>m;
    memset(h,-1,sizeof h);
    while(m--)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    if(spfa())
        puts("Yes");
    else
        puts("No");
        
    return 0;
}

```


