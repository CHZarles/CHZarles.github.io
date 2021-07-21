---
title: Typora+PicGo+Gitee笔记方案
toc: true
copyright: true
comments: true
date: 2020-08-05 13:53:21
keyword:
tags: 工作流
categories: 随笔
top: 
---

### 安装和配置Typora

1. 到官网https://www.typora.io/下载安装即可。

2. 配置图片自动上传：

   1. 偏好设置->图像->插入图片时：选择上传图片。
   2. 偏好设置->图像->上传服务：选择PicGo.app。

3. 使用git将在Gitee上创建的存放笔记的仓库git clone到本地，使用Typora打开目录，之后就可以新增自己的*.md笔记文件，最后使用git add、commit、push上传到Gitee上保存。

   ![image-20200805135717610](https://gitee.com/chzarles/images/raw/master/imgs/image-20200805135717610.png)

### 在Gitee上创建图床仓库

1. 在Gitee上创建存放图片的仓库，比如images4mk。（要设置为公开）

2. 在该项目下创建一个文件夹叫imgs。

3. 在网页上点击头像->设置->私人令牌->生成新令牌（将生成的令牌保存下来）。

   

### 安装PicGo

1. 到github地址https://github.com/Molunerfinn/PicGo/releases下载PicGo软件并安装。

2. 打开PicGo主界面，在插件设置中搜索gitee，安装gitee。安装完之后完成对应配置：

   首先复制Gitee上的图床仓库url，例如：https://gitee.com/jinchengll/images4mk

   1. url：https://gitee.com
   2. ower：填gitee的用户名（对应图床仓库url中的jinchengll）
   3. repo：图床仓库名（对应图床仓库url中的images4mk）
   4. path：imgs
   5. token：填入在2中生成的新令牌字符串。

3. 将默认图床设置为Gitee图床。