#include<string>
#include<fstream>
#include<iostream>
using namespace std;
/*
1.本章讨论的文件I/O相当于控制台I/O，因此仅适用于文本文件。
要创建文本文件，用于提供输入，可使用文本编译器，
如DOS中的EDIT、Windows中的“记事本”和UNIX/Linux系统中的vi或emacs。
也可以使用字处理程序来创建，但必须将文件保存为文本格式。
2.必须在 C++ 源代码文件中包含头文件 <iostream> 和 <fstream>。
*/
int main()
{

	/*Part1:创建文件，并向文件写入内容*/
	ofstream fout;
	fout.open("foutext.txt");
	/*
	1.方法open( )接受一个C-风格字符串作为参数，
	这可以是一个字面字符串，也可以是存储在数组中的字符串。
    2.如果文件存在默认把文件长度截为0，再输出。
	不存在文件就新建文件。
	*/
	
	//将文字写入文件，这里只改变了输出流导向，输入流导向没改变

	fout<<"这是一个测试"<<endl;          //在这里加endl或者'\n'都会导致下面52行(>>输入)的输入出错,用eof输入(利用get()会解决问题
	fout.close();
	
	
	/*Part2*/
	string str="foutext.txt";               //string也可以做形参
	string str1;
	ifstream fin(str); //建立联系
	fin>>str1;         //将文件内容存到字符串，即用文件模拟键盘输入
	cout<<str1;
	fin.close();
	
	
	/*Part3*/
	//void open(const char *filename, ios::openmode mode)
	ofstream ffout("foutext.txt",ios::out|ios::app);      		//写时追加到文件末尾
	ifstream ffin("foutext.txt");
	string str0;
	ffin>>str0;
//	ffin.close();            //断开联系
	cout<<"文件里的内容是:"<<endl<<str0<<endl;;

	cout<<"请把想加入到文件的语句输入:"<<endl;

	cin>>str;

	ffout<<str;                                               //追加到文件末尾
	ffout.close();
	
	ffin.seekg(0);                         					 //*******将流指针移到文件开始。

	char ch;
	str="";             //清空str
	while(ffin.get(ch))
	{
		str+=ch;
	}
	cout<<"文件里的内容是："<<endl<<str;
	
	return 0 ;
}

