---
title: hexo 美化網站收集
date: 2019-10-15 22:53:21
tags: hexo
categories:	随笔
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---

### 有關hexo-next主題的樣式佈置教程

1.基礎官方文檔

 https://theme-next.iissnan.com/getting-started.html 

2.一個很接地氣的知乎教程，頭像設置，標簽，音樂.......都有

 https://zhuanlan.zhihu.com/p/30836436 

3.設置透明度和隨機背景

 hexo（hexo工程文件）-> themes -> next -> source -> css -> _custom  

```css
// 添加背景图片
body {//這裏可以換成本地地址
  background: url(https://source.unsplash.com/random/1600x900?wallpapers);
  background-size: cover;
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-position: 50% 50%;
}

// 修改主体透明度
.main-inner {
  background: #fff;
  opacity: 0.8;
}

// 修改菜单栏透明度
.header-inner {
  opacity: 0.8;
}
```

<!--more-->
其中:

- `background: url()` 中填写的是背景图片的 url 地址, 这里调用了 Unsplash 的 API, 随机选用该网站的高清美图作为博客背景. 该网站所有图片都是免费商用的, 所以无须担心侵权问题;
  网站 API 还有很多有趣的玩法, 参见: Documentation
- `opacity` 指定了对应元素的透明度, 这里是 “0.8”, 可以按需更改.

4.补充

1.[Hexo博客NexT主题从v5.x.x更新到v6.x.x的记录及总结](https://sevencho.github.io/archives/14534beb.html)

2.[增加字数统计](https://co5.me/2018/180613-wordcount.html)

3.[如何github下载单个文件(用于还原补救)](https://www.zhihu.com/question/25369412)

4.[别人的美化系列博客（精）](http://yearito.cn/posts/hexo-advanced-settings.html)

5.[搭建评论+](http://wingjay.com/2017/06/08/rebuild-personal-blog/)

6.[超级全](https://bestzuo.cn/posts/blog-establish.html)

