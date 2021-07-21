---
title: next主题不显示tags和categories的解决方法
date: 2019-10-15 19:58:51
tags: 
	- 解决bug
	- hexo
categories: 
	- 随笔
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

之前不用next主题，现在用回next主题，然后启用了标签栏，发现异常，显示找不到标签，在知乎找到了解决方法。

1.快速输入命令创建一个在source目录下创建tags文件夹(文件夹中包含一个index.md文件）以及类似地创建一个categories文件夹（文件夹中包含一个index.md文件）

` hexo new page tags `

` hexo new page categories `

2.将刚才创建的文件分别编辑成这样



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g7z4w82h7wj30rb083q36.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1g7z59s8c7xj30qs077dg1.jpg)

<!--more-->
附上知乎链接：[hexo 下的分类和表签无法显示，怎么解决？]( https://www.zhihu.com/question/29017171 )

附上连接：[设置默认页面包含categories和tag](https://linlif.github.io/2017/05/27/Hexo%E4%BD%BF%E7%94%A8%E6%94%BB%E7%95%A5-%E6%B7%BB%E5%8A%A0%E5%88%86%E7%B1%BB%E5%8F%8A%E6%A0%87%E7%AD%BE/)

附上终极方案：[解决 Hexo 搭建博客显示不出分类、标签问题](https://blog.csdn.net/wonz5130/article/details/84666519)

