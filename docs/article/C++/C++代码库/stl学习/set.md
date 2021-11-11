```C++
#include<iostream>
#include<stdio.h>
#include<string>
#include<set>
using namespace std;
/*
	//所有元素都是按照字典序自动排序，set只有键值，键值就是市值
	（1）构造：
	（2）通用操作：empty(), insert(), erase(), size(), swap(), find(), count(),
	（3）特有操作：equal_range(), lower_bound(), upper_bound()
*/

void settest()
{
	
	cout<<"-----------------------------this is class set  demo--------------------------" << endl;
	set<string> mySet={"a","b","c"};
	
	/*
		//insert返回一个pair，first是一个迭代器指向具有给定关键字的值，second是一个bool量，若是true则表示插入成功，
		//若为false则表示插入失败，说明已经存在，则该语句什么也不做
		m.insert(e);                                              //插入pair对象
		m.insert(beg, end);                                       //将范围内的元素插入
		m.insert(iter, e);                                        //unknow
	*/
	bool flaf=mySet.empty();									
	auto ret=mySet.insert("d");                                  //插入键值，要是插入相同的元素不知会发生什么
//	mySet.insert("d");											//在插入一遍也没用
	cout << "after inserting, mySet is : \n";
	if (ret.second)
	{
		for (auto tmp : mySet)
			cout << tmp << ends;
	};
	cout<<endl;
	
	auto ret1 = mySet.erase("a");
	cout<<"after eraseing a,mySet is : \n";                         //删除元素
	for (auto tmp : mySet)
	cout << tmp << ends;
	cout<<endl;
	
	auto ret2 = mySet.find("c");                                    //查找元素,找不到返回最后一个元素的指针
	if (ret2 == mySet.end())
		cout << "dont find it \n";
	else
		cout << "find it \n";
	

	unsigned int cnt = mySet.count("d");                            //count只能返回0或1
	printf("d occurred %d times", cnt);

	set<string>::iterator iter_beg = mySet.lower_bound("a");
	set<string>::iterator iter_end = mySet.upper_bound("a");      //给定关键字的范围[lower_bound, upper_bound)


}

int main()
{
	settest();
}

```