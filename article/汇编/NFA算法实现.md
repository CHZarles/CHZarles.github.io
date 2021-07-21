---
title: NFA算法实现
comments: true
copyright: false
toc: true
date: 2020-10-08 10:58:55
tags:
categories:
photo:
top:
cover:
---



## 代码

```python
#文件读入
f=open('input.txt','r',encoding='utf-8')

'''
 problem1: 怎么存储转换矩阵？
 1.要方便查阅: => 用字典
'''

dict = {}
# print("输入两个整数n,m,表示接下来输入的转换矩阵是n行m列的:")
tmp_list = f.readline().split()
n = int(tmp_list[0])
m = int(tmp_list[1])
# print(m,n);
# print("输入矩阵:")
#循环n次
for i in range(n):
    #循环m次
    tmp_list = f.readline().split()


    idx = tmp_list[0]
    #判断键值存不存在
    if dict.get(idx) == None:
        # 给这个状态造一个字典
        dict[idx] = {}
        for j in range(0,m-1):
            dict[idx][str(j)] = []


    for j in range(0, m - 1):
        if tmp_list[j+1] not in dict[idx][str(j)]:
            dict[idx][str(j)].append(tmp_list[j+1])

'''
print(dict)
>>
{
 'x': {'0': ['5','4'], '1': ['-1'], '2': ['-1']}, 
 '5': {'0': ['1'], '1': ['5'], '2': ['5']}, 
 '1': {'0': ['-1'], '1': ['3'], '2': ['4']}, 
 '3': {'0': ['-1'], '1': ['2'], '2': ['-1']},
 '4': {'0': ['-1'], '1': ['-1'], '2': ['2']}, 
 '2': {'0': ['6'], '1': ['-1'], '2': ['-1']}, 
 '6': {'0': ['y'], '1': ['6'], '2': ['6']}
 }
 注：dict['x']['0'] = ['5'，'4'] 表示状态x遇到0代表的字符串时会转成 状态5或4
'''



'''
problem2: 接收输入字母串（字母表）
并且与数字对应起来 =》 同样用字典
要注意的是: 字符串输入的顺序要与矩阵对应的列的含义对应
'''
# print("输入第1列到第m-1列对应的转换字符串(最左边的列是第0列):")
str_list = f.readline().split()
str_dic = {}
for j in range(0,m-1):
    #str_dic[str(j)] = str_list[j]
    str_dic[str_list[j]] = str(j)

'''
input: $ a b
str_dic => {'$': '0', 'a': '1', 'b': '2'}
'''

'''
problem3: 接收初始状态集合和终止状态集合(这个算法好像没用到终止集合)
'''
# print("输入初始状态的集合:")
start = f.readline().split()

'''
problem4: 实现ε闭包
把转换矩阵看出有向图矩阵，使用BFS
输入：一个集合
输出：一个集合
'''


def clouse(queue):
    res = []
    '''
    这是空串闭包运算函数
    queue 是队列(列表)，队列的元素是待进行闭包的集合
    res是运算结果 
    '''
    # BFS求闭包
    #获取空串对应的key,程序认为'$'是空串
    idx = str_dic['$']
    while len(queue)>0:
        #print(res)
        #取出一个状态
        head = queue[0]
        #从队列中删除状态
        queue.pop(0)
        #存入结果
        if head not in res:
            res.append(head)
        #查表获取 head状态 经过一个空串会到达的状态的集合
        #如果这个状态有对应的矩阵行
        if dict.get(head) != None:
            next_state = dict[head][idx]
            for item in next_state:
            #如果没有在res里,存进去
                if item not in res and item != '-1':
                    #print(res,item)
                    res.append(item)
                    queue.append(item)
    return res

# t = clouse(start)
# print(t)


'''
problem5: 实现Ji运算
输入：一个集合
输出：一个集合
'''
def Ji(begin,convert_str):
    '''
    :param begin: 是一个集合，一个待进行J(convert_str)的集合
    :param convert_str: 一个字符串
    :return: 返回一个集合
    '''
    #获取convert_str对应在字典中的key
    idx = str_dic[convert_str]
    #print(idx)
    res = []
    for item in begin:
        #如果该状态有对应行
        if dict.get(item) != None:
            if '-1' in dict[item][idx]:
                dict[item][idx].remove('-1')
            res.extend(dict[item][idx])
            # print(res)
    #去重
    res = list(set(res))
    return res



# tmp = Ji(['x','5','1'],'b')
# print(tmp)
# t = clouse(tmp)
# print(t)


'''
problem6: 造确定化的表
输入：一个集合
输出：一个嵌套字典
'''


def NFA_to_DFA(S0):
    '''
    :param queue: 一个队列，队列里面的元素是集合
    :return: 返回一个字典
    '''
    queue = [S0]
    flag = []
    #开辟一个字典
    res_dic = {}
    while len(queue)>0:
        head = queue[0]
        queue.pop(0)
        #创建新行
        if res_dic.get(str(head)) == None:
            res_dic[str(head)] = {}
            flag.append(set(head))

        #遍历字母表
        for k,v in str_dic.items():
            if k == '$': #空串直接跳过
                continue
            tmp = clouse(Ji(head,k))
            res_dic[str(head)][k] = tmp
            #判断入队
            if res_dic.get(str(tmp)) == None and set(tmp) not in flag:
                queue.append(tmp)

    return res_dic


end = NFA_to_DFA(clouse(start))
for k,v in end.items():
    print(k,':',v)



f.close()
'''
输出
{
"['x', '5', '1']": {'a': ['5', '1', '3'], 'b': ['4', '5', '1']}, 
"['5', '1', '3']": {'a': ['2', '6', '5', '1', '3', 'y'], 'b': ['4', '5', '1']}, 
"['4', '5', '1']": {'a': ['5', '1', '3'], 'b': ['2', '6', '4', '5', '1', 'y']}, 
"['2', '6', '5', '1', '3', 'y']": {'a': ['2', '6', 'y', '5', '1', '3'], 'b': ['4', '6', 'y', '5', '1']}, 
"['2', '6', '4', '5', '1', 'y']": {'a': ['6', 'y', '5', '1', '3'], 'b': ['2', '6', '4', 'y', '5', '1']}, 
"['4', '6', 'y', '5', '1']": {'a': ['6', 'y', '5', '1', '3'], 'b': ['2', '6', '4', 'y', '5', '1']}, 
"['6', 'y', '5', '1', '3']": {'a': ['2', '6', 'y', '5', '1', '3'], 'b': ['4', '6', 'y', '5', '1']}

}

Process finished with exit code 0
'''
```



<!--more-->



## 读入文件

```html
7 4
x  5 -1 -1
5  1  5  5
1 -1  3  4
3 -1  2 -1
4 -1 -1  2
2  6 -1 -1
6  y  6  6
$ a b
x
```












