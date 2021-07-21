---
title: Android入坑指南
comments: true
copyright: false
toc: true
date: 2020-09-18 15:21:45
tags: Android
categories: Android开发入门
photo:
top:
cover:
---

## 全局总览

![image-20200918152320264](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918152320264.png)



这是Android studio整理显示的部分项目文件目录，想快速开发Android一般就编写这些文件就行了

<!--more-->

![image-20200918152517147](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918152517147.png)





.gradle和.idea这两个文件夹由系统自动生成，一般不用管它。



![image-20200918152609735](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918152609735.png)





app目录存放项目需要的代码和资源



![image-20200918152818132](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918152818132.png)

gradle根据配置信息加载进来的东西，不用管

![image-20200918153019817](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918153019817.png)

这个名叫 .gitignore存储很多目录信息，这些目录里的东西不受版本控制

![image-20200918153142687](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918153142687.png)

local.properties这个文件存放sdk的路径

![image-20200918153221897](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918153221897.png)

settings.gradle用来指定项目引入的模块





## App目录下的文件信息

![image-20200918153716410](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918153716410.png)

build目录存放安卓项目编译后产生的文件

libs文件存放第三方的java包

![image-20200918153835260](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918153835260.png)

src文件存放项目源码，图片资源

![image-20200918154006316](https://gitee.com/chzarles/images/raw/master/imgs/image-20200918154006316.png)

build.gradle文件记录当前项目版本信息和依赖的第三方包

## AndroidManifest.xml

**这个文件是Android整个应用程序的配置清单文件，用于向Android提高关于应用的配置信息**

文件包括：包名,组件,权限等信息

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.myapplication">  <!--项目包名-->

      <!--application：包含图标,主题等信息-->
    <application                 
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/AppTheme">
        
         <!--activity：界面信息-->
        <activity android:name=".MainActivity">
             <!--intent-filter：过滤器-->
            <intent-filter>
                 <!--action：行为，这里有个MAIN 说明这个activity是主界面-->
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>

</manifest>
```



**Android四大组件声明标签**

启动一个没有在AndroidManifest.xml文件中声明的组件，会抛出异常

```xml
<activity android:name=" ">   ...... </activity>
<service android:name=" ">   ...... </service>
<provider android:name=" ">   ...... </provider>
<recevier android:name=" ">   ...... </recevier>
```

**权限标签**

```xml
<user-permission android:name="...."> </user-permission>
```



[想了解更多标签信息](https://developer.android.com/guide/topics/manifest/manifest-element)



## 后续开发的参考文档

[官方文档Develper](https://developer.android.com/studio/login)

[民间中文文档](http://www.embeddedlinux.org.cn/androidapi/?utm_source=androiddevtools&utm_medium=website)

[民间论坛](https://www.androiddevtools.cn/)