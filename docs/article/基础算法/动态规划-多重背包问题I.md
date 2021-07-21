---
title: 动态规划-多重背包问题I
date: 2019-10-15 17:19:48
tags: [动态规划, 背包问题] 
categories: 
	- 基础算法 
	- 动态规划
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## [题目描述：多重背包问题I](   https://www.acwing.com/problem/content/4/   )

有 N种物品和一个容量是 V 的背包，每种物品都有无限件可用。

 第 ii 种物品最多有 si 件，每件体积是 vi，价值是 wi。 

求解将哪些物品装入背包，可使这些物品的总体积不超过背包容量，且总价值最大。
输出最大价值。


#### 输入格式

第一行两个整数，N，V，用空格隔开，分别表示物品数量和背包容积。

接下来有 N 行，每行两个整数 vi, wi，si 用空格隔开，分别表示第 i 件物品的体积和价值和数量。

#### 输出格式

<!--more-->
输出一个整数，表示最大价值。

#### 数据范围

0 <N,V≤100
0<vi,wi,si≤100

#### 输入样例

```
4 5
1 2
2 4
3 4
4 5
```

#### 输出样例

```
10
```


## 基本思考框架

![image-20210320085752077](https://gitee.com/chzarles/images/raw/master/imgs/image-20210320085752077.png)



**这个其实和多重背包很类似，具体原理在之前有关动态规划的博客（[01背包](  [https://chzarles.github.io/2019/10/15/%E5%9F%BA%E7%A1%80%E7%AE%97%E6%B3%95/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92-01%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%98/](https://chzarles.github.io/2019/10/15/基础算法/动态规划/动态规划-01背包问题/)  )，[完全背包]( [https://chzarles.github.io/2019/10/15/%E5%9F%BA%E7%A1%80%E7%AE%97%E6%B3%95/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92-%E5%AE%8C%E5%85%A8%E8%83%8C%E5%8C%85%E9%97%AE%E9%A2%98/](https://chzarles.github.io/2019/10/15/基础算法/动态规划/动态规划-完全背包问题/) )）已经讲得很详细了，直接看代码吧：**



#### C++ 代码


```c++
#include<iostream>
using namespace std;
const int N = 110;
int f[N][N];
int v[N],w[N];

int main()
{
    int n,m,v,w,s;
    cin>>n>>m;
    
for(int i = 1 ; i <= n ; i++)
{
 	//在线做法
    cin>>v>>w>>s;
    for(int j = 0 ; j <= m ; j++)
    {
        for(int k = 0 ; k*v<=j&&k <= s ;k ++)
                f[i][j]=max(f[i][j],f[i-1][j-k*v]+k*w);
    }
}
    
    cout<<f[n][m]<<endl;
}
```


