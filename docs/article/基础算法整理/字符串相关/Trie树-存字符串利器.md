---
title: Trie树-存字符串利器
comments: true
copyright: false
date: 2020-01-12 16:38:09
tags:
	- 基础算法
	- Trie
categories:
	- 基础算法
	- 数据结构
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



## Trie简介

摘自:https://songlee24.github.io/2015/05/09/prefix-tree/

**Trie树**，又叫**字典树**、**前缀树（Prefix Tree）**、**单词查找树** 或 **键树**，是一种多叉树结构。如下图：



<!--more-->
![image-20210320091417345](https://gitee.com/chzarles/images/raw/master/imgs/image-20210320091417345.png)





上图是一棵Trie树，表示了关键字集合{“a”, “to”, “tea”, “ted”, “ten”, “i”, “in”, “inn”} 。从上图可以归纳出Trie树的基本性质：

1. 根节点不包含字符，除根节点外的每一个子节点都包含一个字符。
2. 从根节点到**某一个节点**，路径上经过的字符连接起来，为该节点对应的字符串。
3. 每个节点的所有子节点包含的字符互不相同。

通常在实现的时候，会在节点结构中设置一个标志，用来标记该结点处是否构成一个单词（关键字）。

可以看出，Trie树的关键字一般都是字符串，而且Trie树把每个关键字保存在一条路径上，而不是一个结点中。另外，两个有公共前缀的关键字，在Trie树中前缀部分的路径相同，所以Trie树又叫做**前缀树（Prefix Tree）**。



## Trie的优缺点及其应用

Trie树的核心思想是空间换时间，利用字符串的公共前缀来减少无谓的字符串比较以达到提高查询效率的目的。

### 优点

1. 插入和查询的效率很高，都为$O(m)$，其中 $m$ 是待插入/查询的字符串的长度。
   - 关于查询，会有人说 hash 表时间复杂度是$O(1)$不是更快？但是，哈希搜索的效率通常取决于 hash 函数的好坏，若一个坏的 hash 函数导致很多的冲突，效率并不一定比Trie树高。
2. Trie树中不同的关键字不会产生冲突。
3. Trie树只有在允许一个关键字关联多个值的情况下才有类似hash碰撞发生。
4. Trie树不用求 hash 值，对短字符串有更快的速度。通常，求hash值也是需要遍历字符串的。
5. Trie树可以对关键字按**字典序**排序。

### 缺点

1. 当 hash 函数很好时，Trie树的查找效率会低于哈希搜索。
2. 空间消耗比较大。



### 字符串检索

检索/查询功能是Trie树最原始的功能。**思路**就是从根节点开始一个一个字符进行比较：

- 如果沿路比较，发现不同的字符，则表示该字符串在集合中不存在。
- 如果所有的字符全部比较完并且全部相同，还需判断最后一个节点的标志位（标记该节点是否代表一个关键字）。

```
struct trie_node
{
    bool isKey;   // 标记该节点是否代表一个关键字
    trie_node *children[26]; // 各个子节点 
};
```

### 词频统计

Trie树常被搜索引擎系统用于文本词频统计 。

```
struct trie_node
{
    int count;   // 记录该节点代表的单词的个数
    trie_node *children[26]; // 各个子节点 
};
```



思路：为了实现词频统计，我们修改了节点结构，用一个整型变量`count`来计数。对每一个关键字执行插入操作，若已存在，计数加1，若不存在，插入后`count`置1。

**注意：第一、第二种应用也都可以用 hash table 来做。**

### 字符串排序

Trie树可以对大量字符串按字典序进行排序，思路也很简单：遍历一次所有关键字，将它们全部插入trie树，树的每个结点的所有儿子很显然地按照字母表排序，然后先序遍历输出Trie树中所有关键字即可。

### 前缀匹配

例如：找出一个字符串集合中所有以`ab`开头的字符串。我们只需要用所有字符串构造一个trie树，然后输出以`a->b->`开头的路径上的关键字即可。

trie树前缀匹配常用于搜索提示。如当输入一个网址，可以自动搜索出可能的选择。当没有完全匹配的搜索结果，可以返回前缀最相似的可能。

### 作为其他数据结构和算法的辅助结构

如后缀树，AC自动机等。



## 写法模板

```c++
int son[N][26], cnt[N], idx;
// 0号点既是根节点，又是空节点
// son[][]存储树中每个节点的子节点
// cnt[]存储以每个节点结尾的单词数量


/* 
son数组尽量开大一点吧,怕不够用。
可以把son数组看成一片连续的虚拟内存。 
idx其实就是这个虚拟内存的指针 。

然后在插入字符串时，每新出现一个分支，
就要新开辟一片内存，对应下面代码的  
if (!son[p][s]) son[p][s] = ++idx;（其实++idx改成++(++idx)也能过，因为idx只是用来指向新内存的，不过很浪费空间而已）

cnt[i]存储 在idx这个内存位置结束的字符串 有多少个。 
*/ 

// 插入一个字符串
void insert(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;
}

// 查询字符串出现的次数
int query(char *str)
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}

作者：yxc
链接：https://www.acwing.com/blog/content/404/
来源：AcWing
```





## [例题：Trie字符串统计](https://www.acwing.com/problem/content/837/)

```c++
//AC code
#include<iostream>
#include<cstring>
using namespace std;
const int N = 100010;
char str[N];

//Mem主存数组。第一维是指针维度(所以第一维很大),第二维是该指针指向的字母。例如,Mem[idx][a-'a']表示idx指针指向的字母a的指针。
//我也不知道第一维是N,够不够大
int Mem[N][26],idx = 0;
//cnt[i] = 1 ， 表示idx = i 这个位置是某个单词的结尾
int cnt[N];

void insert(char str[])
{
    //p指向"父字母"的指针
    int p = 0;
    for(int i = 0 ; str[i] ; i++)
    {
        int u = str[i] - 'a';
        //存在
        if(Mem[p][u] == 0) Mem[p][u] = ++idx ;
        p = Mem[p][u];
    }
    cnt[p] ++ ;
}

int search(char str[])
{
    int p = 0;
    for(int i = 0 ; str[i] ; i++)
    {
        int u = str[i] - 'a';
        //存在
        if(Mem[p][u]==0)
        {
          //  puts("check");
            return 0;
        }
        else
            p = Mem[p][u];
    }
    if(cnt[p])
        return cnt[p];
}


int main()
{
    int N ;
    cin>>N;
    while(N--)
    {
        char op[2];
        cin>>op>>str;
        //cout<<str<<endl;
        if(op[0]=='I')
            insert(str);
        else
            cout<<search(str)<<endl;
    }
    return 0;
}
```



## [例题：最大异或对](https://www.acwing.com/problem/content/description/145/)

```c++
//AC code
#include<iostream>
#include<algorithm>
using namespace std;
const int N = 100010*31;
//存数的"字符串"
int str[32];
int ans = -1;
//trie
int Mem[N][2] , idx = 0;
//节点存数
int cnt[N];

//十进制转二进制字符串
void to_bin(int x)
{
    for(int i = 0 ; i < 31 ; i++ )
    {
        str[i] = x%2;
        x = x/2;
    }
}

//存进Trie数
void insert(int str[],int num)
{
    int p = 0;
    //高位在树顶
    for(int i = 31 ; i >= 0 ; i-- )
    {
        int u = str[i];
        if(Mem[p][u]==0) Mem[p][u] = ++idx;
        p = Mem[p][u];
    }
    cnt[p] = num;
}

//查询最佳匹配操作数
int search(int str[])
{
    int p = 0;
    
     for(int i = 31 ; i >= 0 ; i-- )
    {
        int u = str[i];
        if( Mem[p][!u]!= 0 ) 
            p = Mem[p][!u];
        else
            p = Mem[p][u]; 
    }
    return cnt[p];
}


int main()
{
    int n,num;
    cin>>n;
    while(n--)
    {
        cin>>num;
        to_bin(num);
        ans = max(ans,search(str)^num);
        insert(str,num);
    }
    cout<<ans<<endl;
    return 0;
}
```



------

 