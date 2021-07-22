---
title: 为什么Dijkstra算法不能求负权图的最短路?
comments: true
copyright: false
date: 2019-12-09 00:11:09
tags:
	- 随笔
	- 算法
categories:
	- 基础算法
	- 图论
photo: https://wiki.mbalib.com/w/images/6/65/Dijkstra%E7%AE%97%E6%B3%95%E5%9B%BE.jpg
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



## [引用Stackoverflow上的一个问题](https://stackoverflow.com/questions/6799172/negative-weights-using-dijkstras-algorithm)

I am trying to understand why Dijkstra's algorithm will not work with negative weights. Reading an example on [Shortest Paths](http://www.ics.uci.edu/~eppstein/161/960208.html), I am trying to figure out the following scenario:

```
    2
A-------B
 \     /
3 \   / -2
   \ /
    C
```

<!--more-->
From the website:

> Assuming the edges are all directed from left to right, If we start with A, Dijkstra's algorithm will choose the edge (A,x) minimizing d(A,A)+length(edge), namely (A,B). It then sets d(A,B)=2 and chooses another edge (y,C) minimizing d(A,y)+d(y,C); the only choice is (A,C) and it sets d(A,C)=3. But it never finds the shortest path from A to B, via C, with total length 1.

I can not understand why using the following implementation of Dijkstra, d[B] will not be updated to `1` (When the algorithm reaches vertex C, it will run a relax on B, see that the d[B] equals to `2`, and therefore update its value to `1`).

```
Dijkstra(G, w, s)  {
   Initialize-Single-Source(G, s)
   S ← Ø
   Q ← V[G]//priority queue by d[v]
   while Q ≠ Ø do
      u ← Extract-Min(Q)
      S ← S U {u}
      for each vertex v in Adj[u] do
         Relax(u, v)
}

Initialize-Single-Source(G, s) {
   for each vertex v  V(G)
      d[v] ← ∞
      π[v] ← NIL
   d[s] ← 0
}

Relax(u, v) {
   //update only if we found a strictly shortest path
   if d[v] > d[u] + w(u,v) 
      d[v] ← d[u] + w(u,v)
      π[v] ← u
      Update(Q, v)
}
```



## 高票答案

The algorithm you have suggested will indeed find the shortest path in this graph, but not all graphs in general. For example, consider this graph:

![Figure of graph](https://i.stack.imgur.com/rmowk.png)

Assume the edges are directed from left to right as in your example,

Your algorithm will work as follows:

1. **First, you set `d(A)` to `zero` and the other distances to `infinity`.**
2. **You then expand out node `A`, setting `d(B)` to `1`, `d(C)` to `zero`, and `d(D)` to `99`.**
3. **Next, you expand out `C`, with no net changes.**
4. **You then expand out `B`, which has no effect.**
5. **Finally, you expand `D`, which changes `d(B)` to `-201`.**

Notice that at the end of this, though, that `d(C)` is still `0`, **even though the shortest path to `C` has length `-200`.** Your algorithm thus fails to accurately compute distances in some cases. Moreover, even if you were to store back pointers saying how to get from each node to the start node `A`, you'd end taking the wrong path back from `C` to `A`.



## 补充

题主那个图刚好可以用Dijkstra求出来，但高票答案提供的图，就不能用Dijkstra求出来了。

  **题主的例子，最后全部点都是橙色(算法运行结果正确)：**

![dj1.gif](http://ww1.sinaimg.cn/large/006eb5E0gy1g9pshxisg4g30ap08kdig.gif)

​	        

 **高票的例子，最后有一个点红色(那个点没被正确更新，算法运行结果错误)**

![dj.gif](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1g9psjiax4pg30bl08xac7.gif)



​									        







------

 