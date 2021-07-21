---
title: 模拟散列表
tags:
  - 数据结构
  - 散列表
categories:
  - 基础算法
  - 数据结构
comments: true
copyright: false
date: 2020-01-14 11:45:00
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：模拟散列表](  https://www.acwing.com/problem/content/842/ )

维护一个集合，支持如下几种操作：

1. “I x”，插入一个数x；
2. “Q x”，询问数x是否在集合中出现过；

现在要进行N次操作，对于每个询问操作输出对应的结果。


#### 输入格式

<!--more-->
第一行包含整数N，表示操作数量。

接下来N行，每行包含一个操作指令，操作指令为”I x”，”Q x”中的一种。

#### 输出格式

对于每个询问指令“Q x”，输出一个询问结果，如果x在集合中出现过，则输出“Yes”，否则输出“No”。

每个结果占一行。

#### 数据范围

 1 ≤N≤ 1e5
−1e9≤x≤1e9

#### 输入样例

```
5
I 1
I 2
I 3
Q 2
Q 5
```

#### 输出样例

```
Yes
No
```




## 算法/复杂度

简单提要：

![image-20200114123515656](C:\Users\123456\AppData\Roaming\Typora\typora-user-images\image-20200114123515656.png)



拉链法

```c++
(1) 拉链法
    //其实就是邻接表
    int h[N], e[N], ne[N], idx;

    // 向哈希表中插入一个数
    void insert(int x)
    {
        //让K>0
        int k = (x % N + N) % N;
        e[idx] = x;
        ne[idx] = h[k];
        h[k] = idx ++ ;
    }

    // 在哈希表中查询某个数是否存在
    bool find(int x)
    {
        int k = (x % N + N) % N;
        for (int i = h[k]; i != -1; i = ne[i])
            if (e[i] == x)
                return true;

        return false;
    }
```



开放寻址法

```c++
(2) 开放寻址法
    int h[N];

    // 如果x在哈希表中，返回x的下标；如果x不在哈希表中，返回x应该插入的位置
    int find(int x)
    {
        int t = (x % N + N) % N;
        while (h[t] != null && h[t] != x)
        {
            t ++ ;
            if (t == N) t = 0;
        }
        return t;
    }

```



## C++ 代码


```c++
#include <cstring>
#include <iostream>

using namespace std;

const int N = 200003, null = 0x3f3f3f3f;

int h[N];

int find(int x)
{
    int t = (x % N + N) % N;
    while (h[t] != null && h[t] != x)
    {
        t ++ ;
        if (t == N) t = 0;
    }
    return t;
}

int main()
{
    memset(h, 0x3f, sizeof h);

    int n;
    scanf("%d", &n);

    while (n -- )
    {
        char op[2];
        int x;
        scanf("%s%d", op, &x);
        if (*op == 'I') h[find(x)] = x;
        else
        {
            if (h[find(x)] == null) puts("No");
            else puts("Yes");
        }
    }

    return 0;
}


```


