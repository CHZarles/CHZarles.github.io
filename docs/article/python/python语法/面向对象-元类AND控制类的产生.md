---
title: 面向对象-元类AND控制类的产生
comments: true
copyright: false
toc: true
date: 2020-08-18 10:12:24
tags:
categories: python基本语法
photo:
top:
cover:
---



## 是什么是元类

直接看demo

```python
class OldboyTeacher(object):
    school='oldboy'

    def __init__(self,name,age):
        self.name=name
        self.age=age

    def say(self):
        print('%s says welcome to the oldboy to learn Python' %self.name)
        

t1=OldboyTeacher('egon',18)
print(type(t1)) #查看对象t1的类是<class '__main__.OldboyTeacher'>

print(type(OldboyTeacher)) # 结果为<class 'type'>，证明是调用了type这个元类而产生的OldboyTeacher，即默认的元类为type
```

<!--more-->

![img](https://gitee.com/chzarles/images/raw/master/imgs/1036857-20180715004832944-116819652.png)







## Class关键字创建类的流程分析

**上文我们基于python中一切皆为对象的概念分析出：我们用class关键字定义的类本身也是一个对象，负责产生该对象的类称之为元类（元类可以简称为类的类），内置的元类为type**

**class关键字在帮我们创建类时，必然帮我们调用了元类OldboyTeacher=type(...)，那调用type时传入的参数是什么呢？必然是类的关键组成部分，一个类有三大组成部分，分别是**

**1、类名class_name='OldboyTeacher'**

**2、基类们class_bases=(object,)**

**3、类的名称空间class_dic，类的名称空间是执行类体代码而得到的**

```python
#Class 生成类的流程演示
class_name = "Oldboy_Teacher"
#2.类的基类
class_bases = (object,)
#3.执行类体代码
class_dic = {}
class_body = """

def __init__(self,name,age):
    self.name = name
    self.age = age

def say(self):
    print('%s %s ' %(self.name,self.age))

"""

exec(class_body,{},class_dic)#将运行代码产生的变量扔进名称空间，并放进字典class_dic
print(class_dic)#{'__init__': <function __init__ at 0x03941100>, 'say': <function say at 0x039B4FA0>}

# 4.调用元类（只有这步是我们可以控制的）
Oldboy_Teacher = type(class_name,class_bases,class_dic)
print(Oldboy_Teacher)# <class '__main__.Oldboy_Teacher'>

obj = Oldboy_Teacher('alex','30')
obj.say() #alex 30 
```

## 自定义元类    控制   类OldboyTeacher   的生成

**一个类没有声明自己的元类，默认他的元类就是type，除了使用内置元类type，我们也可以通过继承type来自定义元类，然后使用metaclass关键字参数为一个类指定元类**



其实就是改最后一句` People = type(class_name,class_bases,class_dic)` ==》`People = Mymeta(class_name,class_bases,class_dic)`
**只要用class定义类时 显示指定参数 metaclass = Mymeta 然后自己再定义Mymeta就行**

调用Mymeta会发生三件事
1.首先造一个空对象=>People （调用Mymeta下的\__new\__函数,如果没定义就用系统默认的)   (从必要性来说 其实到这步就ok了 后面的步骤都是锦上添花)
2.调用Mymeta这个类的\__init\__方法，初始化对象
3.返回对象

### Demo 0

```python
'''
一个类没有声明自己的元类，默认他的元类就是type，除了使用内置元类type，我们也可以通过继承type来自定义元类，然后使用metaclass关键字参数为一个类指定元类
'''
class Mymeta(type): #只有继承了type类才能称之为一个元类，否则就是一个普通的自定义类
    pass

class OldboyTeacher(object,metaclass=Mymeta): # OldboyTeacher=Mymeta('OldboyTeacher',(object),{...})
    school='oldboy'

    def __init__(self,name,age):
        self.name=name
        self.age=age

    def say(self):
        print('%s says welcome to the oldboy to learn Python' %self.name)
```



### Demo 1 

```python
'''
利用__init__函数定制自己想要的功能
自定义元类可以控制类的产生过程，类的产生过程其实就是元类的调用过程,即OldboyTeacher=Mymeta('OldboyTeacher',(object),{...})，调用Mymeta会先产生一个空对象OldoyTeacher，然后连同调用Mymeta括号内的参数一同传给Mymeta下的__init__方法，完成初始化，于是我们可以
'''
class Mymeta(type): #只有继承了type类才能称之为一个元类，否则就是一个普通的自定义类
    def __init__(self,class_name,class_bases,class_dic):
        # print(self) #<class '__main__.OldboyTeacher'>
        # print(class_bases) #(<class 'object'>,)
        # print(class_dic) #{'__module__': '__main__', '__qualname__': 'OldboyTeacher', 'school': 'oldboy', '__init__': <function OldboyTeacher.__init__ at 0x102b95ae8>, 'say': <function OldboyTeacher.say at 0x10621c6a8>}
        super(Mymeta, self).__init__(class_name, class_bases, class_dic)  # 重用父类的功能

        if class_name.islower():
            raise TypeError('类名%s请修改为驼峰体' %class_name)

        if '__doc__' not in class_dic or len(class_dic['__doc__'].strip(' \n')) == 0:
            raise TypeError('类中必须有文档注释，并且文档注释不能为空')

class OldboyTeacher(object,metaclass=Mymeta): # OldboyTeacher=Mymeta('OldboyTeacher',(object),{...})
    """
    类OldboyTeacher的文档注释
    """
    school='oldboy'

    def __init__(self,name,age):
        self.name=name
        self.age=age

    def say(self):
        print('%s says welcome to the oldboy to learn Python' %self.name)
```



## 自定义元类    控制   类OldboyTeacher   的调用

**调用一个对象，就是触发对象所在类中的call方法的执行，如果把OldboyTeacher也当做一个对象，那么在OldboyTeacher这个对象的类中也必然存在一个call方法**





```
class Foo:
    def __call__(self, *args, **kwargs):
        print(self)
        print(args)
        print(kwargs)

obj=Foo()
#1、要想让obj这个对象变成一个可调用的对象，需要在该对象的类中定义一个方法__call__方法，该方法会在调用对象时自动触发
#2、调用obj的返回值就是__call__方法的返回值
res=obj(1,2,3,x=1,y=2)
```

由上例得知，调用一个对象，就是触发对象所在类中的__call__方法的执行，如果把OldboyTeacher也当做一个对象，那么在OldboyTeacher这个对象的类中也必然存在一个\__call\__方法

```python
class Mymeta(type): #只有继承了type类才能称之为一个元类，否则就是一个普通的自定义类
    def __call__(self, *args, **kwargs):
        print(self) #<class '__main__.OldboyTeacher'>
        print(args) #('egon', 18)
        print(kwargs) #{}
        return 123

class OldboyTeacher(object,metaclass=Mymeta):
    school='oldboy'

    def __init__(self,name,age):
        self.name=name
        self.age=age

    def say(self):
        print('%s says welcome to the oldboy to learn Python' %self.name)



# 调用OldboyTeacher就是在调用OldboyTeacher类中的__call__方法
# 然后将OldboyTeacher传给self,溢出的位置参数传给*，溢出的关键字参数传给**
# 调用OldboyTeacher的返回值就是调用__call__的返回值
t1=OldboyTeacher('egon',18)
print(t1) #123
```



> 上例的\__call\__相当于一个模板，我们可以在该基础上改写__call__的逻辑从而控制调用OldboyTeacher的过程，比如将OldboyTeacher的对象的所有属性都变成私有的

```python
class Mymeta(type): #只有继承了type类才能称之为一个元类，否则就是一个普通的自定义类
    def __call__(self, *args, **kwargs): #self=<class '__main__.OldboyTeacher'>
        #1、调用__new__产生一个空对象obj
        obj=self.__new__(self) # 此处的self是类OldoyTeacher，必须传参，代表创建一个OldboyTeacher的对象obj

        #2、调用__init__初始化空对象obj
        self.__init__(obj,*args,**kwargs)

        # 在初始化之后，obj.__dict__里就有值了
        obj.__dict__={'_%s__%s' %(self.__name__,k):v for k,v in obj.__dict__.items()}
        #3、返回初始化好的对象obj
        return obj

class OldboyTeacher(object,metaclass=Mymeta):
    school='oldboy'

    def __init__(self,name,age):
        self.name=name
        self.age=age

    def say(self):
        print('%s says welcome to the oldboy to learn Python' %self.name)

t1=OldboyTeacher('egon',18)
print(t1.__dict__) #{'_OldboyTeacher__name': 'egon', '_OldboyTeacher__age': 18}
```

## 属性查找

[详见](https://www.cnblogs.com/linhaifeng/articles/8029564.html)