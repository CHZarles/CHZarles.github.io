---
title: 重装系统恢复hexo博客
comments: true
copyright: false
date: 2020-02-09 11:02:24
tags:
categories:
	- 随笔
photo:
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



今天偶然想到要更新博客了，写文章时发现用不了。原来是电脑重装了系统，Hexo博客需要重新部署恢复。具体该怎么弄呢，就在网上搜索解决办法，网上的内容也是挺乱的。自己就按照某些思路来尝试解决。

还好，没费多大功夫，Hexo博客又重生了。

![img](http://ww1.sinaimg.cn/large/66101050ly1fd2szawfzuj20hs07sq36)

## Hexo博客部署恢复

先说下我的情况：电脑重装了系统，hexo博客文件夹还在D盘存着，完好无损。现在就是借助这些文件恢复部署一下就好了。具体方法如下：

**一、**安装 node.js 和 git for windows

<!--more-->
**二、**配置 git 个人信息，生成新的 ssh 密钥：

```
git config --global user.name "helloqingfeng"
git config --global user.email "2538745263@qq.com"
ssh-keygen -t rsa -C "2538745263@qq.com"
```

你需要把邮件地址和用户名换成你自己的，然后一路回车，使用默认值即可。

如果一切顺利的话，可以在用户主目录里找到.ssh目录，里面有id_rsa和id_rsa.pub两个文件，打开id_rsa.pub文件（打开为txt就行），复制里面的内容。

**三、**将生成的ssh公钥（刚复制的内容）复制到Github的settings里面的ssh选项里去。（参考GitHub教程：[新增 SSH 密钥到 GitHub 帐户](https://help.github.com/cn/articles/adding-a-new-ssh-key-to-your-github-account)）

**四、**安装Hexo：

这步要在blog文件夹里

```
npm install hexo-cli -g
```

**五、**打开原来的hexo博客所在文件夹，只需保留_config.yml，theme/，source/，scaffolds/，package.json，.gitignore 这些项目，删除其他的文件。

**六、**然后打开 git bush 运行命令：

这步要在blog文件夹里

```
npm install
```

**七、**安装部署插件：

这步要在blog文件夹里

```
npm install hexo-deployer-git --save
```

**八、**接下来直接hexo g hexo d试一下是否成功。

这步要在blog文件夹里

```
hexo g
hexo d
```

博客部署恢复完成，可以打开网站看一下。

**九、**发表新文章：

新建 Markdown 文件及文章标题：

```
hexo new post “文章名字”
```

然后使用 Markdown 软件编辑内容，

最后上传更新文章：

```
hexo g
hexo d
```



新思路：https://segmentfault.com/q/1010000013246651























------

 