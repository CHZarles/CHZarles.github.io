```C++
#include<algorithm>//std::make_heap, std::pop_heap, std::push_heap, std::sort_heap
#include<iostream>
#include<vector>

using namespace std;

/*
STL中并没有把heap作为一种容器组件，heap的实现亦需要更低一层的容器组件（诸如list,array,vector）作为其底层机制。
Heap是一个类属算法，包含在algorithm头文件中。
在STL中为堆heap的创建和操作提供了4种算法：make_heap，pop_heap，push_heap和sort_heap。

http://www.cplusplus.com/reference/algorithm/make_heap/
*/
void heap_text()
{
	 int a[] = {15, 1, 12, 30, 20};
    vector<int> ivec(a, a+5);
    for(vector<int>::iterator iter=ivec.begin();iter!=ivec.end();++iter)
        cout<<*iter<<" ";
    cout<<endl;

    make_heap(ivec.begin(), ivec.end());//建堆
    for(auto iter=ivec.begin();iter!=ivec.end();++iter)
        cout<<*iter<<" ";
    cout<<endl;

    pop_heap(ivec.begin(), ivec.end());//先pop,然后在容器中删除
    ivec.pop_back();
    for(vector<int>::iterator iter=ivec.begin();iter!=ivec.end();++iter)
        cout<<*iter<<" ";
    cout<<endl;

    ivec.push_back(99);//先在容器中加入，再push
    push_heap(ivec.begin(), ivec.end());
    for(vector<int>::iterator iter=ivec.begin();iter!=ivec.end();++iter)
        cout<<*iter<<" ";
    cout<<endl;

    sort_heap(ivec.begin(), ivec.end());
    for(vector<int>::iterator iter=ivec.begin();iter!=ivec.end();++iter)
        cout<<*iter<<" ";
    cout<<endl;


}

int main()
{
	heap_text();
	return 0;
}

```