```C++
/*list是一个线性双向链表结构，它的数据由若干个节点构成，每一个节点都包括一个信息块（即实际存储的数据）,
一个前驱指针和一个后驱指针。它无需分配指定的内存大小且可以任意伸缩，这是因为它存储在非连续的内存空间中，
并且由指针将有序的元素链接起来。由于其结构的原因，list 随机检索的性能非常的不好，
因为它不像vector 那样直接找到元素的地址，而是要从头一个一个的顺序查找，这样目标元素越靠后，它的检索时间就越长。
检索时间与目标元素的位置成正比。虽然随机检索的速度不够快，但是它可以迅速地在任何节点进行插入和删除操作。
因为list 的每个节点保存着它在链表中的位置，插入或删除一个元素仅对最多三个元素有所影响，
不像vector 会对操作点之后的所有元素的存储地址都有所影响，这一点是vector 不可比拟的。

list特点 
(1) 不使用连续的内存空间这样可以随意地进行动态操作；
(2) 可以在内部任何位置快速地插入或删除，当然也可以在两端进行push 和pop 。
(3) 不能进行内部的随机访问，即不支持[ ] 操作符和vector.at() ；
(4) 相对于verctor 占用更多的内存。
*/


/*
	(1)构造：
	(2)通用函数：push_back(), pop_back(), empty(), clear(), swap(), insert(), erase(), reverse()
	(3)特有操作：push_front(), pop_front(), merge(), remove(), remove_if()
*/

#include<string>
#include<list>
#include<iostream>
#include<algorithm> 
using namespace std;
bool is_odd(const int x)	//删除偶数 
{
	return (x%2==0);
} 

int main()
{
	cout << "-----------------------------this is class List demo--------------------------" << endl;
	
	//几种构造函数 
	list<int> mylist0{3,5};
	list<int> mylist9(mylist0);				//复制构造函数 
	list<int> mylis8(3,5);					//初始化为含三个数据为5的结点
	list<int> mylist(mylist0);
	
	
	
	mylist.push_back(7);
	mylist.push_front(1);					//双向链表支持在两端的快速插入和删除
	cout<<"after push ,my list is : "<<endl;
	for(auto tmp: mylist)
	{
		cout<<tmp<<ends;
	}	
	cout<<endl;
	
	auto ret=mylist.insert(mylist.end(),9);			//插入操作和其它一样
	cout<<"after inserting , mylist is : "<<endl;
	 for(auto tmp: mylist)
	{
		cout<<tmp<<ends;
	}	
	cout<<endl;
	
	list<int> mylist1={2,4,6,8};
	mylist.merge(mylist1);					//合并两个有序链表，且元素无重复，返回合并后的链表(有序的)，若二者其一就会出错
	cout<<"after merging , my list is : "<<endl; 
	for(auto tmp: mylist)
	{
		cout<<tmp<<ends;
	}	
	cout<<endl;
	
	mylist.remove(8);						//删除所有满足8的元素 
	cout << "after removing, myList is : " << std::endl;
	for (auto tmp : mylist)
	{
	cout << tmp << ends;
	}
	cout << endl;

	mylist.remove_if(is_odd);				//remove_if() //当条件满足自己定义的is_odd时满足 
	cout << "after removing_if(is_odd), myList is : " << endl;
	for (auto tmp : mylist)
	{
	cout << tmp << ends;
	}
	cout << endl;
	
	reverse(mylist.begin(),mylist.end());
	cout << "after reverse, myList is : " << endl;
	for (auto tmp : mylist)
	{
	cout << tmp << ends;
	}
	cout << endl;
} 

```