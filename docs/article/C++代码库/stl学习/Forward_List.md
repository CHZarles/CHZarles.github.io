```C++
#include<iostream>
#include<string>
#include<forward_list>      //单向链表
#include<iterator>          //其中的advance函数，可以移动迭代器移动指定长度
/*
	（1）构造：
	（2）通用函数：empty(), front(）, swap(), clear(), push_front(), pop_front(), reverse(),
	（3）特有函数：inert_after(), erase_after(), before_begin(), remove(), remove_if(), unique(), sort(), merge(),
*/
using namespace std;
void myForward_list()
{
	cout << "------------------------this is class forward_list demo-----------------------" << endl;
	forward_list<int> myforward_list={1,3,7,5,5};
	cout<<"the front of myforward_list is: "<<myforward_list.front()<<endl;
	
	myforward_list.push_front(0);
	for(auto tmp:myforward_list)
	cout<<tmp<<ends;
	cout<<endl;
	
	/*
	lst.insert_after(p, t)       //在迭代器p之后的位置插入元素t，返回指向插入元素的迭代器
	lst.insert_after(p, b, e)    //在迭代器p之后插入范围为[b, e）的元素，返回最后一个插入链表的迭代器
	*/
	
	auto iter=myforward_list.before_begin();
	advance(iter,1);                                                	 //另外还可以使用iterator中的advance函数对迭代器进行偏移
	myforward_list.insert_after(myforward_list.before_begin(),9);       //插入元素9//list 和forward_list虽然不支持+,-操作，但是支持++， （--）操作
	cout << "after inserting , myForward_list is : " << endl;
	for (auto iter = myforward_list.begin(); iter != myforward_list.end(); ++iter)
	{
		cout << *iter << ends;
	}
	cout << endl;
	
	
	myforward_list.erase_after(myforward_list.before_begin());          //在迭代器p之后的位置插入元素t，返回指向插入元素的迭代器
	cout<<"after erasing , myforward_list is :"<<endl;
	for(auto iter=myforward_list.begin();iter!=myforward_list.end();++iter)
	cout<<*iter<<ends;
	cout<<endl;
	
	//myForward_list.remove(9);                                                    //删除某一特定值元素
	//myForward_list.remove_if(is_odd);                                             //按照传入谓词来删除某一元素
	
	myforward_list.unique();                                                        //踢出重复元素
	cout<<"after unique , myforward_list is :"<<endl;
	for (auto iter = myforward_list.begin(); iter != myforward_list.end(); ++iter)
	{
		cout << *iter << ends;
	}
	cout << '\n';
	
	
	myforward_list.sort();                                                              //对链表数据进行排序
	cout<<"after sorting ,myforward_list is :"<<endl;
	for (auto iter = myforward_list.begin(); iter != myforward_list.end(); ++iter)
	{
		cout << *iter << ends;
	}
	cout << '\n';
	
	
}

int main()
{
	myForward_list();
	
	
}

```