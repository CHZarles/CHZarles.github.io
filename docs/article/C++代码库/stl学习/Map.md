```C++
#include <iostream>
#include <string>
#include <map>
#include <iterator>
using namespace std;
/*
	（1）构造：std::map<string, int>, std::pair<string, int>, std::make_pair(v1, v2)
	（2）特殊操作：first, second, insert(), erase(), count(), find(), operator[]();
*/
void myMapTest()
{
	cout << "----------------------------this is class map demo---------------------------" << endl;
	map<string,int> myMap;
	
	auto mypair1=pair<string,int>("hello",1);
	auto mypair2=make_pair("world",10);
	
	/*
		//insert返回一个pair，first是一个迭代器指向具有给定关键字的值，second是一个bool量，若是true则表示插入成功，
		//若为false则表示插入失败，说明已经存在，则该语句什么也不做，只能手动的++
		m.insert(e);                                          //插入pair对象
		m.insert(beg, end);                                   //将范围内的元素插入
		m.insert(iter, e);                                    //unknow
	*/

	auto ret1=myMap.insert(mypair1);
	if (ret1.second)    //插入成功
		cout << "insert successfully\n";
	printf("%s---->%d\n", ret1.first->first.c_str(), ret1.first->second);
	
	auto ret2=myMap.insert(mypair2);
	if (ret2.second)
		cout << "insert successfully\n";
	printf("%s---->%d\n", ret2.first->first.c_str(), ret2.first->second);


	++myMap["word"];                                            //使用下标运算符，若不存在，则创建新的键值对(word, 0)
	 cout << "after operator[], myMap is : \n";
	for (auto iter = myMap.begin(); iter != myMap.end(); ++iter)
	{
		printf("%s---->%d\n", iter->first.c_str(), iter->second);
	}
	/*
		m.erase(k);                             //删除关键字为k的元素
		m.erase(p);                             //删除迭代器p指向的元素
		m.erase(b, e);                          //删除范围内的元素
	*/
	myMap.erase("word");
	cout << "after erasing, myMap is : \n";
	for (auto iter = myMap.begin(); iter != myMap.end(); ++iter)
	{
		printf("%s---->%d\n", iter->first.c_str(), iter->second);
	}
	

	unsigned cnt = myMap.count("hello");                                            //返回关键字的数量
	printf("hello occurred %d times\n", cnt);
	std::cout << "find keyword hello : \n";
	auto ret3 = myMap.find("hello");         //返回关键字的迭代器,如果找不到，返回myMap.end();
	printf("%s---->%d\n", ret3->first.c_str(), ret3->second);

}
int main()
{
	
	myMapTest();
}

```