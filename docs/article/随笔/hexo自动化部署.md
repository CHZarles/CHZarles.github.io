---
title: hexo自动化部署
toc: true
copyright: true
comments: true
date: 2020-08-05 17:00:48
keyword:
tags:
categories: 随笔
top:
---



## 步骤一 自动打开Typora

找到存放博客的根目录文件夹 

我这里是 `C:\Users\Charles\blog`

然后新建一个`scripts` 文件夹 ，进入文件夹，新建一个`javascrip`文件，命名为newpost

```javascript
//C:\Users\Charles\blog\scripts\newpost
//将下面的 C:\\Program Files\\Typora\\bin\\typora.exe 改成自己电脑的 typora.exe的路径

var spawn = require('child_process').exec;
hexo.on('new', function(data){
  spawn('start  "C:\\Program Files\\Typora\\bin\\typora.exe" ' + data.path);
});
```

布置完的效果是 在命令行输入`hexo n xxx` ，脚本自动打开对应的md文件。

<!--more-->

## 步骤二 设置Typora自动上传图片到图床

详细步骤看 : {% post_link 随笔/Typora-PicGo-Gitee笔记方案 %}



## 步骤三 一键更新Gitee page

参考:[利用python实现自动签到及giteePages自动部署](https://gocos.cn/2020/02/6.html)



## 步骤四 集成操作

自己写个带GUI界面的小软件，集成一下。

原理就是代码调用cmd输入hexo指令

![image-20200805181057165](https://gitee.com/chzarles/images/raw/master/imgs/image-20200805181057165.png)

相关博文: {% post_link 随笔/Markdown文件Post-Front-matter的tag的批处理 %}