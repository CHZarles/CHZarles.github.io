---
title: Markdown文件Post Front-matter的tag的批处理
comments: true
copyright: true
date: 2020-03-08 21:07:41
tags:
	- python
categories:
	- 随笔
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gcmttxkmmmj313g0jogpr.jpg
toc: true
---





## 增加Front-matter的tag种类 

**源代码**

```python
#增加标签
def addtag(filepath,tagname,word):
    with open(filepath ,encoding = 'utf-8' , mode = 'r') as f:
        article = ""
        while True:
            line = f.readline()
            if len(line)==0:
                break
            else:
                article+=line
        
        #定位,因为有两个--- 定位下面那个
        idx = article.find('''---''',3)

        #构造字符串co
        newtag = tagname + ": " + word + "\n"
        new_article = article[0:idx] + newtag + article[idx:]
        
    # 重新写入文件
    with open(filepath ,encoding = 'utf-8' , mode = 'w') as fw:
        fw.write(new_article)
        #print(article)



tagname=input("输入tagname:")
word=input("输入参数:")

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        if os.path.join(root, name)[-3:] == ".md" :
            print("work")
            print(os.path.join(root, name))
            addtag(os.path.join(root, name),tagname,word)

    for name in dirs:
        #print(os.path.join(root, name))
        #print(os.path.join(root, name)[-4:-1])
        if os.path.join(root, name)[-3:] == ".md":
            print("work")
            addtag(os.path.join(root, name),tagname,word)

```



## 修改Front-matter的tag的参数

这个参考这个项目：https://github.com/acupt/hexoez

## readme

批量修改指定目录中md文件的属性，如tags,categories,titlec

```
python hexoez.py <op> <matter_name> [<argv>[,]]  <file>
op: add / del / update 
matter_name: 可以是标题，也可以是普通项
```

### 增加tag

给tmp.md的tags增加元素newTag1 newtag2 newtag3

```
python hexoez.py add tags newTag1 newtag2 newtag3 source/tmp.md
```

给指定目录(包括子目录)下所有md文件增加tag，其他命令的批量操作类似，最后的参数不是.md结尾默认为目录

```
python hexoez.py add tags newTag1 newtag2 newtag3  source/
```

### 删除tag

删除标签Tag1 tag2 tag3

```
python hexoez.py del tags Tag1 tag2 tag3 source/tmp.md
```

### 修改tag

```
python hexoez.py update tags old_tag new_tag source/tmp.md
```

### 修改标题

```
python hexoez.py update title old_title new_title source/tmp.md
```



## 批量添加\<!--more-->以及copyright置false

```python
import os

# 增加<!--more-->以及将处理post_modify，处理copyright
def modify(filepath):

    with open(filepath ,encoding = 'utf-8' , mode = 'r') as f:
        article = ""
        # 数行数
        tmp = 0
        #两个用来检测代码块的标记
        flag_1 = False
        flag_2 = False
        checked = False
        while True:
            tmp +=1
            checked = False
            line = f.readline()

            #检测到第一个
            if "```" in line and flag_2 == False and flag_1 == False :
                flag_1 = True
                print("检测到第一个,行"+str(tmp)+" :"+line)
                checked= True

            #检查到第二个
            if "```" in line and flag_1 == True and flag_2 == False and checked == False:
                flag_2 = True
                flag_1 = False
                print("检测到第二个,行"+str(tmp)+" :"+line)

            #经过了第二个标记，清空标记
            if not ("```" in line) and flag_1 == False and flag_2 == True :
                flag_2 = False
                print("清空标记，行"+str(tmp)+" :" + line)

            if len(line) == 0:
                break
            article += line

            if tmp > 25 and flag_1 == False and flag_2 == False:
                article += "<!--more-->\n"
                #锁住
                flag_2 = True
                flag_1 = True


        # 获取要修改的位置
        idx1 = article.find('''Post modified:''')
        idx2 = article.find('''copyright: true''')
        # 处理post_modify
        if idx1 != -1:
            new_article = ""
            new_article = article[0:idx1]
            article = new_article

        # 处理copyright
        if idx2 != -1:
            article = article.replace('''copyright: true''' ,'''copyright: false''' ,1)

        # 重新写入文件
        with open(filepath ,encoding = 'utf-8' , mode = 'w') as fw:
            fw.write(article)
            #print(article)



for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        if os.path.join(root, name)[-3:] == ".md" :
            print("work")
            print(os.path.join(root, name))
            modify(os.path.join(root, name))

    for name in dirs:
        #print(os.path.join(root, name))
        #print(os.path.join(root, name)[-4:-1])
        if os.path.join(root, name)[-3:] == ".md":
            print("work")
            modify(os.path.join(root, name))

```



## 根据文本内容搜索文件路径

```python
import os

def findit(filepath,keyword):
    article = ""
    with open(filepath ,encoding = 'utf-8' , mode = 'r') as f:
        article = ""
        while True:
            line = f.readline()
            if len(line)==0:
                break
            else:
                article+=line
        
    #如果找到 输出文件路径
    idx = article.find(keyword)
    if(idx!=-1): print(filepath)


keyword = input("输入关键字：")

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        if os.path.join(root, name)[-3:] == ".md" :
            print("work")
            findit(os.path.join(root, name),keyword)

    for name in dirs:
        #print(os.path.join(root, name))
        #print(os.path.join(root, name)[-4:-1])
        if os.path.join(root, name)[-3:] == ".md":
            print("work")
            findit(os.path.join(root, name),keyword)
```





## 提醒

**各位最好自己先试验一下这些代码再用！**

**各位最好自己先试验一下这些代码再用！**

**各位最好自己先试验一下这些代码再用！**

**做好备份再用！**

**做好备份再用！**

**做好备份再用！**

有问题留言:D