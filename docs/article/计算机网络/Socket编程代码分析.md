---
title: Socket编程代码分析
comments: true
copyright: false
date: 2020-06-19 16:14:07
tags: socket
categories: 计算机网络
photo:
top:
cover: http://ww1.sinaimg.cn/large/006eb5E0gy1gfxrlpnxchj30fo0c10t3.jpg
toc: true
---



## 面向连接的套接字系统调用时序图（C/S结构）

编程时按照这个顺序就好，要注意，系统在调用**accept()，recv()**的时候是阻塞的，一直等待客户端发来消息



![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxohvpcmtj30ex0fx77a.jpg)

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxrhwncvrj30oa0evn2l.jpg)

<!--more-->

## 简单通信程序的演示（单线程，CS结构，面向连接-TCP/IP）

Winsocket**编程手册**：https://docs.microsoft.com/en-us/windows/win32/api/_winsock/

### **服务器代码**


```c++
#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h> 
#include <stdio.h>
#include <iostream>

// Need to link with Ws2_32.lib
//让程序链接这个库 
#pragma comment(lib, "ws2_32.lib")
using namespace std;

int  main()
{  
    WORD wVersionRequested;//版本字 
    WSADATA wsaData;//存版本信息 
    int err;
    wVersionRequested = MAKEWORD(2, 2);
    err = WSAStartup(wVersionRequested, &wsaData);//初始化API
    if (err != 0) {                                  
        printf("WSAStartup failed with error: %d\n", err);
        return 1;
    }
    
    //调用socket创建套接字, socket返回套接字描述符  
	SOCKET server = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
	
	//设置套接字地址
	 sockaddr_in  serveraddr;
	 //自己设置地址族,端口,IP 
	 serveraddr.sin_family = AF_INET; 
	 serveraddr.sin_port = htons(12345);
	 //本机地址是 10.3.63.456
	 serveraddr.sin_addr.s_addr = inet_addr("10.3.63.456");
	 
    //套接字地址和服务器绑定起来
    /*强制转换的原因 
	struct sockaddr
	数据结构用做bind、connect、recvfrom、sendto等函数的参数，指明地址信息。
	但一般编程中并不直接针对此数据结构操作，而是使用另一个与sockaddr等价的数据结构 struct sockaddr_in
	sockaddr_in和sockaddr是并列的结构，指向sockaddr_in的结构体的指针也可以指向
	sockadd的结构体，并代替它。也就是说，你可以使用sockaddr_in建立你所需要的信息,
	在最后用进行类型转换就可以了关于这两个结构体晚上内容很多核心就是
	一般编程中并不直接针对此数据结构操作，而是使用另一个与sockaddr等价的数据结构 
	struct sockaddr_in
    */ 
	int iResult = bind(server, (sockaddr *) &serveraddr, sizeof (serveraddr));
	 if (iResult == SOCKET_ERROR) {
        wprintf(L"bind failed with error %u\n", WSAGetLastError());
        closesocket(server);
        WSACleanup();
        return 1;
    }
    else
        wprintf(L"bind returned success\n");
	
	while(1) 
	{ 
	//接下来监听
	listen(server,5);
		
   	//设置数据结构保存客户端口的信息	
    sockaddr_in caddr;
    int ll = sizeof(caddr);
    
    //addrlen引用的整数最初包含addr指向的空间量。在返回时，它将包含返回的地址的实际字节长度。
    //调用了accept函数,如果没有请求连接，进程会一直处于阻塞状态 
    SOCKET client = accept (server, (sockaddr*) &caddr, &ll) ;
    cout<<"有一个客户机连接进来了"<<endl; 
    char recvdata[1024] = {0};
    //(描述符,buffer指针,buffer长度,标志位) 
    recv (client,recvdata,1023, 0); 
    cout<<"客户机发来指令: "<<recvdata<<endl;
    
    //发生反馈信息
    char sendbuff[100] = "收到消息";
	iResult = send(client, sendbuff, (int)strlen(sendbuff), 0 );
    if (iResult == SOCKET_ERROR) {
        wprintf(L"send failed with error: %d\n", WSAGetLastError());
        closesocket(client);
        WSACleanup();
        return 1;
    } 
    //关闭连接 
    closesocket(client);
    } 
    WSACleanup();

}
```


### **客户端代码**


```c++
#include <winsock2.h> 
#include <stdio.h>
#include <iostream>

// Need to link with Ws2_32.lib
//让程序链接这个库 
#pragma comment(lib, "ws2_32.lib")

using namespace std;


int  main()
{
    WORD wVersionRequested;//版本字 
    WSADATA wsaData;//WSAStartup()函数返回信息到这个数据结构 
    int err;//初始化情况 
    wVersionRequested = MAKEWORD(2, 2);
    err = WSAStartup(wVersionRequested, &wsaData);
    if (err != 0) 
	{                             
        printf("WSAStartup failed with error: %d\n", err);
        return 1;
    }
    
    //调用socket创建套接字, socket返回套接字描述符,AF_INET表示IPv4查手册得到相关信息 
	SOCKET client = socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);

	//构造要连接的服务器的名叫sockaddr_in的数据结构  
	sockaddr_in serveraddr; 
	serveraddr.sin_family = AF_INET; 
	serveraddr.sin_port = htons(12345);
	serveraddr.sin_addr.s_addr = inet_addr("10.3.63.456");
	
	//客户端连接服务器 
	cout<<"尝试连接服务器"<<endl; 
	int iResult = connect(client, (SOCKADDR *) & serveraddr, sizeof (serveraddr));
    if (iResult == SOCKET_ERROR) {
        wprintf(L"connect function failed with error: %ld\n", WSAGetLastError());
        iResult = closesocket(client);
        if (iResult == SOCKET_ERROR)
            wprintf(L"closesocket function failed with error: %ld\n", WSAGetLastError());
        WSACleanup();
        return 1;
    }
    cout<<"连接成功"<<endl;
    
    //发数据 
    char sendbuf[100]; 
    char order = 'N'; 
    while(order!='Y')
	{
		memset(sendbuf,0,sizeof(sendbuf));
	    cout<<"输入你想对服务器发送的信息：";
		cin>>sendbuf;
		cout<<"是否要发送(Y/N)? " ;
		cin>>order;
		 
	 }
	 
	//返回发送的字节数 
    iResult = send( client, sendbuf, (int)strlen(sendbuf), 0 );
    if (iResult == SOCKET_ERROR) {
        wprintf(L"send failed with error: %d\n", WSAGetLastError());
        closesocket(client);
        WSACleanup();
        return 1;
    }
    printf("Bytes Sent: %d\n", iResult);


    // Receive until the peer closes the connection
    int recvbuflen = 1024;
    char recvbuf[1024] = "";
    do {

        iResult = recv(client, recvbuf, recvbuflen, 0);
        if ( iResult > 0 )
           {
           	  cout<<"服务器反馈:"<<recvbuf<<endl; 
           	  wprintf(L"Bytes received: %d\n", iResult);
			} 
        else if ( iResult == 0 )
            wprintf(L"Connection closed\n");
        else
            wprintf(L"recv failed with error: %d\n", WSAGetLastError());

    } while( iResult > 0 );


    // close the socket
    iResult = closesocket(client);
    if (iResult == SOCKET_ERROR) {
        wprintf(L"close failed with error: %d\n", WSAGetLastError());
        WSACleanup();
        return 1;
    }

	
}
```


### **接下来我要演示一下阻塞**

1.**现在只运行服务器程序**

程序调用accept()，一直阻塞等待客户端连接到服务器

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxpzw6pxfj30y003s3yh.jpg)

2.**接下来运行客户端程序**

客户端的窗口显示连接到了服务器

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxrkh4r1dj30x803zjry.jpg)

服务器窗口也显示"有一个客户端连接到服务器" , 这时accept()的阻塞解除了，系统阻塞在recv()函数

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxrkpfa6ij30xc03rdg7.jpg)

3.**接下来客户端发送信息**

客户端发送完信息后就结束程序了

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxqk4h15jj30xn08tq3l.jpg)

服务器程序又回到了accept()阻塞

![image.png](http://ww1.sinaimg.cn/large/006eb5E0gy1gfxqkh5iqij30xz05pt8v.jpg)



## 简单的多人聊天室(多线程，C/S结构，面向连接-TCP/IP)

**代码出处:https://mp.weixin.qq.com/s/GbumgW7uLFugDCKEDj1Jzw**

**Python Socket 编程详细介绍:https://gist.github.com/kevinkindom/108ffd675cb9253f8f71**

### 服务端程序(无界面)


```python
import socket
from threading import Thread #多线程

#接着创建一下 socket ，绑定地址和端口号：
host = '10.3.63.456'
port = 12345
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

#我们可以定义字典来存放用户的数据，比如连接用户的昵称以及地址：

client = {}
addresses = {}

#再来定义下服务器可接收的 client 连接数：

accept_num = 10


#接着我们来实现一下用户的消息处理方法，我们可以接收用户发来的昵称消息，这时候就可以在聊天室里面进行广播，告诉大家 “xxx 加进来了”，另外我们可以把用户的昵称加到字典中来：

def handle_client_in(conn, addr):
    nikename = conn.recv(1024).decode('utf8')
    print("进入handle_clint_in")
    
    welcome = f'\n欢迎 {nikename} 加入聊天室\n'
    client[conn] = nikename
    brodcast(bytes(welcome, 'utf8'))

    #接下来可以定义一个 While 循环，来监听用户发送的消息，当服务端获取到用户发来的消息之后，我们可以在聊天室进行广播，告诉大家 “xxx 发来了 xxx 消息”，而当用户由于异常而退出聊天室的时候，我们可以将连接关掉，并且把字典存着的用户数据给删掉：

    while True:
       try:
          msg = conn.recv(1024)
          brodcast(msg, nikename+':')
       except:
          conn.close()
          del client[conn]
          brodcast(bytes(f'{nikename} 离开聊天室', 'utf8'))

#那么如何对聊天室的用户进行广播呢，因为我们刚刚在字典中都存储了连接进来的用户连接，那么就可以通过循环的方式向每个用户发送消息：

def brodcast(msg, nikename=''):
    for conn in client:
       conn.send(bytes(nikename, 'utf8') + msg)

#接下来可以在 main 方法在监听用户的连接：

if __name__ == '__main__':
    s.listen(accept_num)
    print('服务器已经开启，正在监听用户的请求..')

      #接着可以写一个 whie 循环来接收用户的连接：

    while True:
         conn, address = s.accept()
         print(address, '已经建立连接')
         conn.send('欢迎你来到异世界，请输入你的昵称:'.encode('utf8'))

#接收到用户的连接之后，我们就可以获取到用户的连接和地址信息，可以把地址保存到我们刚刚定义的字典里面来：

         addresses[conn] = address

#要支持多个用户的信息收发，我们可以开启线程：

         Thread(target=handle_client_in, args=(conn, address)).start()

```




### 客户端程序(有界面)


```python
from tkinter import * 
import socket
from threading import Thread

Host='10.3.63.456'
Port = 12345
S =socket.socket(socket.AF_INET,socket.SOCK_STREAM)
S.connect((Host, Port))

def send():
        #在send 方法中获取输入框中的内容，发送给 socket ，然后再清空输入框中的内容：
        print("已发送")
        send_msg= text_text.get('0.0 ',END )
        S.send(bytes(send_msg,' utf8') )
        text_text.delete( '0.0',END)
        
      
root = Tk()
root.title('chatroom')


message_frame = Frame(root,width=480, height=300,bg='black')
text_frame = Frame(root,width=480, height=100)
send_frame =Frame (root,width=480,height=30)



text_message = Text(message_frame,bg='azure')
text_text = Text(text_frame)
button_send= Button(send_frame,text= 'send' )
#绑定触发器,下面要定义send方法
button_send=Button(send_frame,text='发送',command=send)

message_frame.grid( row=0, column=0, padx=3,pady=6) 
text_frame.grid( row=1,column=0,padx=3, pady=6)
send_frame.grid( row=2,column=0 )


message_frame.grid_propagate(0)
text_frame.grid_propagate(0)
send_frame.grid_propagate(0)



text_message.grid( )
text_text.grid( )
button_send.grid( )
def get_msg():
        while True:
            try:
                print("受到消息")
                msg = S.recv(1024).decode( 'utf8')
                
                text_message.insert(END, msg)
            except:
                break

#必须放在loop里
receive_thread =Thread(target=get_msg)
receive_thread.start() 
root.mainloop()   

```


## 简单通信程序的演示（单线程，CS结构，面向无连接-UDP/IP）

面向无连接和面向连接的区别是，前者发送消息前要绑定套接字，发送只用send()方法就好，后者无需绑定套接字，但每次发消息都要指定(IP,端口)，要用sendto()方法

**服务器程序**

```python
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 绑定端口:
s.bind(('127.0.0.1', 9999))
#创建Socket时，SOCK_DGRAM指定了这个Socket的类型是UDP。绑定端口和TCP一样，但是不需要调用listen()方法，而是直接接收来自任何客户端的数据：
print('Bind UDP on 9999...')
while True:
    # 接收数据:
    data, addr = s.recvfrom(1024)
    print('Received from %s:%s.' % addr)
    s.sendto(b'Hello, %s!' % data, addr)
```

**客户端程序**

```python
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for data in [b'Michael', b'Tracy', b'Sarah']:
    # 发送数据:
    s.sendto(data, ('127.0.0.1', 9999))
    # 接收数据:
    print(s.recv(1024).decode('utf-8'))
s.close()
```

