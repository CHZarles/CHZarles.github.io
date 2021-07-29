```C++
#include<iostream>
#include<cstring>
#include<vector>
#include <fstream>
using namespace std;
int main()
{
	/*利用循环输入文本*/
	//这个代码模式适合EOF为结束的输入。具体细节，视系统而定
	/*每次读取一个字符，直到遇到EOF的输入循环的基本设计如下：*/
	ifstream fcin;
	fcin.open("test.txt",ios::in);
	string str;
	vector<char> str1;
	char ch;
	while(fcin.get(ch))          				//cin会在需要bool值得地方自动转换，检测到eof，自动返回false，此外这是最通用的设计
	{
		
		//str1.push_back(ch);

								//cin.get(ch)储存了换行
	
		str+=ch;
	}
	
	//检测default位
	if(fcin.fail())								//判断是不是eof
	cout<<"检测到EOF"<<endl;
	fcin.close();
    											//如果键盘模拟的eof之后还有其它输入，加上cin.clear(),这里用的是文件流不需要加


    /*
    ofstream ffout("keyword.txt",ios::out|ios::app);  
    ffout<<str<<endl; 
    */
    
    cout<<"文本内容是："<<endl<<str<<ends;
    
    
}

```