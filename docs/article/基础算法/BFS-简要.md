---
title: BFS 简要
comments: true
copyright: false
date: 2019-10-23 16:07:12
tags: [基础算法,图论]
categories: [基础算法,图论]
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## Bfs的过程

以有向图的广搜来理一理广搜的一些概念。

![bfs1.gif](https://gitee.com/chzarles/images/raw/master/imgs/ZhgjITqXFvNlxfo.gif)



**注：广搜应该也有对应的广搜树，不过很少讨论。**

## BFS实现

BFS一般用队列实现（大家都知道吧）

这个地址有详细步骤（~~我太懒了~~）。
<!--more-->

[论执着与稳重的好处----DFS/BFS](https://www.acwing.com/blog/content/461/)

## c++模板

写广搜一般是这样思路：

reference:[论执着与稳重的好处----DFS/BFS](https://www.acwing.com/blog/content/461/)

```c++
queue<int> q;
st[0] = true; // 表示1号点已经被遍历过
q.push(0);

while (q.size())
{
    int t = q.front();
    q.pop();

    for (int i = h[t]; i != -1; i = ne[i])
    {
        int j = e[i];
        if (!s[j])
        {
            st[j] = true; // 表示点j已经被遍历过
            q.push(j);
            ......
        }
    }
}
```



## 一些性质

**对比BFS和DFS：**

![image.png](https://gitee.com/chzarles/images/raw/master/imgs/mPpW9Zk148BRIti.jpg)

------

 