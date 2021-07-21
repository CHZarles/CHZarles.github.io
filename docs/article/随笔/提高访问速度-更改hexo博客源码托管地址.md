---
title: 提高访问速度-更改hexo博客源码托管地址
comments: true
copyright: false
date: 2020-03-06 14:34:00
tags:
	- hexo
categories: 
	- 随笔
photo: http://ww1.sinaimg.cn/large/006eb5E0gy1gck7fr4cvwj30yg09s0tk.jpg
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



### 前言

本来我的博客的访问地址是 chzarles.github.io的，但github的访问速度太慢了。恰好了解到国内有个类似的网站gitee看能不能把网页托管在那上面。

试了一下，成功了哈。

reference：[Hexo+码云+git快速搭建免费的静态Blog](https://segmentfault.com/a/1190000016083265)

<!--more-->

### 步骤

**1.登录码云，新建一个项目。**

> 如果想以`http://chzarles.gitee.io`这种一级域名的形式访问bolg，那么我们需要建立一个与自己个性地址同名的项目，如 `https://chzarles.com/chzarles` 这个用户，在创建项目时**项目名称**应该为`chzarles`。



**2.在克隆/下载中，复制HTTPS里的链接。**

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gck7oati43j30u40bstbg.jpg)



​    如果界面是这样的，复制这个链接

​     ![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gck7fr4cvwj30yg09s0tk.jpg)



**3.找到Blog本地的根目录里的_config.yml文件，找到deploy这一项，将之前复制的HTTPS里的链接复制到repo后面，然后修改type的值为git，然后保存即可。**

![undefined](http://ww1.sinaimg.cn/large/006eb5E0gy1gck7pbansnj30kh0gvjs6.jpg)



**4部署Hexo到码云**

就像我们平常部署到github一样

```
$ hexo deploy
```



**5.使用码云page服务**

选择`服务—>Gitee Pages`，我们使用master分支，然后直接点“启动”，即可启动page服务。
更多关于码云page的说明可参考官网：[码云Gitee帮助文档](http://git.mydoc.io/?t=154714)

![undefined](http://ww1.sinaimg.cn/large/006eb5E0gy1gck7plfu61j30up0ep76g.jpg)





**6.配置SSH**

有了免费的服务器(page)之后，我们还可以把它和我们个人电脑作一个绑定，以后使用git通讯就不用总是输入账号密码，自然方便多了！

![undefined](http://ww1.sinaimg.cn/large/006eb5E0gy1gck7px6cacj30ui0h4myl.jpg)

这里不啰嗦了，最完美的配置SSH教程参见官网：[生成并部署SSH key](http://git.mydoc.io/?t=154712)





最后网站部署就完成了！！！！！！！