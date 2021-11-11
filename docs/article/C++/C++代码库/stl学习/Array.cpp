//#array  执行效率比<vector>高
#include <iostream>
#include <array>
#include<vector>
#include<algorithm>
using namespace std;
struct bird
{
	int a,b;
};
int main ()
{
//******************************************************初始化template <class T，size_t N> class array, T是数据类型（泛型），后面是长度******************************
	
	/*int一维数组的初始化*/
	

	array<int ,5> temp0={1,2,3};    //剩下的初始化为0
	array<int ,5> temp2={0};        //全部初始化为0
	array<int ,5> temp1;            //元素值是随机的
	array<int ,5> tempa=temp0;      //复制构造函数.长度要一致
	
	
	/*int二维数组初始化*/
	array<array<int,2>,3 >temp3={0};   //全部初始化为0,如果是myarray2D={1}，则只有myarray2D[0][0]为1,其它为0
	

//*********************************************************迭代器Iterators，用auto获取迭代器（本质是指针）******************************************************
	//begin	 Return iterator to beginning
	//end	 Return iterator to end
	for (auto it = temp0.begin(); it != temp0.end(); ++it)
	{
        std::cout << *it << ends;
	}
	cout<<endl;
	//rbegin	Return reverse iterator to reverse beginning
	//rend  	Return reverse iterator to reverse end
	for (auto it = temp0.rbegin(); it != temp0.rend(); ++it)
	{
        std::cout << *it << ends;
	}
	cout<<endl;

	//下面的函数都返回常迭代器
	//cbegin	Return const_iterator to beginning
	//cend		Return const_iterator to end
	//crbegin	Return const_reverse_iterator to reverse beginning
	//crend		Return const_reverse_iterator to reverse end
	
//***********************************************************容量类Capacity*******************************************************************



	//size	Return size
	//max_size	Return maximum size

	for(int i=0;i<temp0.size();i++)         //size()和max_size()好像没区别;
	cout<<temp0[i]<<ends;
	cout<<endl;
	
	//empty	Test whether list is empty,没卵用的操作，array怎么会空
	if(temp0.empty()) cout<<"数组已空"<<endl;
	else cout<<"数组没空"<<endl;
	
	
//*****************************************************************访问类操作*********************************************************************


	std::array<int, 5> arr = {1, 2, 3, 4, 5};
	
	//索引访问的两种方法
    cout << "array[0] = " << arr[0] << endl;
    cout << "array.at(4) = " << arr.at(4) << endl;
    
   //越界测试
   /* std::cout << "array[0] = " << arr[8] << std::endl;            //如果是arr[8]，这里输出了0
    std::cout << "array.at(4) = " << arr.at(8) << std::endl;        //如果是arr.at(8)，会把错误信息输出到屏幕
    */
    
    //front	Access first element
    cout << "array.front() = " << arr.front() << endl;
    //back	Access last element
    cout << "array.back() = " << arr.back() << endl;
    //data	Get pointer to first data
    cout << "&array: " << arr.data() << " = " << &arr << endl;


//************************************************************整体操作*************************************************

	//fill	Fill array with value
	array<int, 5> arr0;
    arr0.fill(5);  // fill
    cout << "array values: ";
    for (auto i : arr0)
	{
        cout << i << " ";
    }
    cout << endl;

    //swap	Swap content，只能整体交换，而且两者长度一致,只测试了int
    array<int, 3> first = {1, 2, 3};
    array<int, 3> second = {6, 5, 4};
    cout << "first  array values: ";
    for (auto it = first.begin(); it != first.end(); ++it)
	{
        cout << *it << " ";
    }
    cout << endl;

    cout << "second array values: ";
    for (auto it = second.begin(); it != second.end(); ++it)
	 {
        cout << *it << " ";
    }
    cout << endl;

	// first.swap(second);
	//array里的函数 swap,效果和algorithm里的swap一样可以整体替换,但algorithm的swap()可以换单个元素
	swap(first[2],second[1]);
    cout << "swap array success!" <<endl;
    cout << "first  array values: ";
    for (auto it = first.begin(); it != first.end(); ++it)
	{
        cout << *it << " ";
    }
    cout << endl;
    
    cout << "second array values: ";
    for (auto it = second.begin(); it != second.end(); ++it)
	{
        cout << *it << " ";
    }
    cout << endl;

//*************************************************************Compare**************************************************************

	array<int,5> a = {10, 20, 30, 40, 50};
    array<int,5> b (a);
    array<int,5> c = {50, 40, 30, 20, 10};

    if (a == b) {
        cout << "a == b" << endl;
    } else {
        cout << "a != b" << endl;
    }

    if (a == c) {
        cout << "a == c" << endl;
    } else {
        cout << "a != c" << endl;
    }

    if (a < c) {
    	cout << "a < c" << endl;
    } else {
        cout << "a >= c" << endl;
    }
//****************************************************************other******************************************
	//get( array)	Get element (tuple interface)
	//tuple_element<array>	Tuple element type for array
	//tuple_size<array>	Tuple size traits for array
    //不常用，等到会元组之后再填补相关内容

    return 0;
}


