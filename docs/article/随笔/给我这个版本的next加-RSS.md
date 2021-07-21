---
title: 给我这个版本的next加 RSS
date: 2019-10-17 18:28:56
tags: 
	- hexo 
	- next
photo: 
categories: 随笔
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

## 安装插件

```javascript
npm install hexo-generator-feed
```

## 在主題配置文件中（主題的＿config.yml）

```javascript
找到(没有就在末尾加)
#extention的部分
plugin:
  - hexo-generator-feed

在這個文件中可以找到（好像不加也可以）
rss:（這裏空下來就行，詳情可以看它原本的註解）
```

<!--more-->
## 在hexo配置文件中

```
#Directory 中
加入
feed:
  type: atom
  path: atom.xml
  limit: 20
```

## reference

- [韩海龙blog Hexo—正确添加RSS订阅](http://hanhailong.com/2015/10/08/Hexo—正确添加RSS订阅/)
- [参考原理](https://abelsu7.top/2018/03/07/hexo-rss/#%E5%8F%82%E8%80%83%E6%96%87%E7%AB%A0)

