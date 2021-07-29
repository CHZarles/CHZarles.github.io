#include<iostream>
#include<cstring>
using namespace std;
int main()
{
	
	
	
	int year;
	char name[20];
	cout<<"你爱我爱了多少年？"<<endl;
	(cin>>year).get();                  //没有这个get(),后面geiline没有输入机会（如果用换行结束year的输入的话）
	cout<<"我最喜欢的电影是什么？";
	cin.getline(name,20);
	cout<<"对，就是"<<name;
	
	
	
	
	
}
