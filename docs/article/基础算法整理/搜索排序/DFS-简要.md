---
title: DFS 简要
comments: true
copyright: false
date: 2019-10-23 14:25:16
tags: [基础算法,图论]
categories: [基础算法,图论]
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



## dfs的过程

以有向图的深搜来理一理深搜的一些概念。

![dfs1.gif](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1g886drw4u4g30ap08kq82.gif)

**注：图中灰色的边表示这条边指向的点之前已经被访问了。**

**注: 深搜不仅适用于图，深搜又叫暴搜（其实就是穷举）。**



奇妙的是，每个深搜的过程，都会对应一颗搜索树，上面那个图对应的搜索树是这样的：
<!--more-->

![dfs2.gif](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1g886hshvgag30ap08k0wm.gif)



## 怎么思考dfs

![image.png](https://gitee.com/chzarles/images/raw/master/imgs/006eb5E0gy1g8931tr8m3j30xg0fstc1.jpg)

思考深搜要注意三个地方

- “顺序”：就是要找一种搜索顺序，能把各种情况都枚举出来**（想想上面说到的搜索树，每一步对应一个节点以及其延伸出的子树）**。
- “回溯” ：回溯说白了就是在一个dfs内，结束了调用的dfs，回到原来的进程。一般回溯都回到原来的进程后，都要“恢复现场” 。记住 , 一但你从一个深搜出来 ， 就马上恢复现场，**就像事情没发生过一样**。当然，不一定所有题目都要恢复现场。（深搜无模板，看题目而定）
- “剪枝”：剪枝就是把不必要的搜索进程砍掉（emmm，这就很抽象了，视题目而定）



## c++模板

写深搜一般是这样思路：

reference:[论执着与稳重的好处----DFS/BFS](https://www.acwing.com/blog/content/461/)

```c++
dfs()//参数用来表示状态  
{  
    if(到达终点状态)  
    {  
        ...//根据题意添加  
        return;  
    }  
    if(越界或者是不合法状态)  
        return;  
    if(特殊状态)//剪枝
        return ;
    for(扩展方式)  
    {  
        if(扩展方式所达到状态合法)  
        {  
            修改操作;//根据题意来添加  
            标记；  
            dfs（）；  
            (还原标记)；  
            //是否还原标记根据题意  
            //如果加上（还原标记）就是 回溯法  
        }  

    }  
}
//大佬所言：就像你去学习一个混合运算，你要必须学习加法思想，然后减法，再是乘除。
```

------

 