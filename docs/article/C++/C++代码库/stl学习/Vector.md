```C++
 #include<vector>
#include<iostream>
#include<string>
#include<algorithm>
using namespace std;
bool cmp1(double m, double n);
int main()
{
/**************************************************初始化**********************************************************/
/*常见操作：size(),empty(),push_back,pop_back(),insert(),erase(),clear(),swap()*/

    //一，初始化
	vector<int> temp0={1,2,3,4,5};
	vector<double> 	temp1(5,0.11);
	vector<double> 	temp2(temp1.begin()+1,temp1.begin()+3);  	//取指针范围内的数初始化，左闭右开
	vector<double>  temp3(temp1);                      			//复制构造函数
	vector<double>  temp4=temp1;                                //重载了“= ”
	
	for(auto temp:temp3)
	cout<<temp<<ends;
	cout<<endl;
	
	vector<string> temp5={"i love you","wsq"};                 //泛型，初始化，有两个元素的string数组，相当于二维的char
	vector<vector<string>> temp6(5,temp5);                      //泛型，string二维数组,相当于三维的char，不过第三维不整齐
	
	for(int i=0;i<5;++i)
	{ 	for(int j=0;j<2;++j)
		  cout<<temp6[i][j]<<ends;
		  
		cout<<endl;
	}
	
/************************size()*************************************/
	//输出temp6长度
	cout<<"size of temp6 is "<<temp6.size()<<endl;

/***********************empty()和pop_back()和push_back()和earse()和clear()**************************************************/

	//使用clear(), 清空temp6
	temp6.clear();
	if(temp6.empty()) cout<<"temp6 is empty"<<endl;
	temp6.push_back(temp5);
	if(!temp6.empty()) 
	cout<<"operation of push_back success"<<endl<<"Its content is:";
	cout<<temp6.at(0).at(0)<<ends<<temp6.at(0).at(1)<<endl;         	//用at()访问
	
	
	/*
	erase()版本：
		（1）erase(p)         //删除迭代器p所指元素，返回下一个元素的迭代器
		（2）erase(b, e)      //删除迭代器[b, e) 范围内的元素；
		                      //关于迭代器确定范围都是左闭右开！！！！
	*/
	
	//使用erase删除范围元素(实参是迭代器)
	temp6.erase(temp6.begin(),temp6.end())   ;               		//这里直接把整个string数组的数组删掉了
	if(temp6.empty()) cout<<"Eraser worked , temp6 is empty"<<endl;
	
	//使用erase删除单个元素
	temp6.push_back(temp5);                                         //先插入元素temp5

   	//只删除了string数组里的第一个元素"i love you"删掉了,返回值是下一个元素的迭代器
	auto it=temp6[0].erase(temp6[0].begin());
	cout<<"present in the way of at():"<<temp6.at(0).at(0)<<endl<<"presrnt int the way of pointer:"<<*it<<endl;
   	

   	//使用pop_back
   	temp6.pop_back();
	if(temp6.empty()) cout<<"operation of push_back success "<<"temp6 is empty"<<endl;
	
	
	/*
	insert()版本：
		（1）insert(p, t)     //在迭代器p之前创建一个值为t，返回指向新添加的元素的迭代器
		（2）insert(p, b, e)  //将迭代器[b, e）指定的元素插入到p所指位置，返回第一个插入元素的迭代器
		（3）insert(p, il)    //将列表中的元素插入，返回第一个插入元素的迭代器
		                      //关于迭代器确定范围都是左闭右开！！！！
	*/
	
	cout<<"before insert temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	for(int i=0;i<2;i++ )
	temp1.insert(temp1.begin()+2,6);
	
	cout<<"After insert temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	temp1.clear();
	temp1.insert(temp1.begin(),6,1.2);
	cout<<"After insert temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	temp1.insert(temp1.begin()+1,temp2.begin(),temp2.end());
	cout<<"After insert temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;

/*****************************************************************************resize()************************************************/

	temp1.resize(20,7);//将temp1容量调至20，多则删，少则补7；
	cout<<"After resize temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
/*****************************************************************************swap********************************************************/
	cout<<"before swap with temp4 temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	temp1.swap(temp4);//长度不一样也无所谓
	
	cout<<"After swap with temp4 temp1's element:";
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
/********************************************************************algoriothm****************************************************************/

	//排序
	temp1.clear();
	for(int i=1;i<10;i++)
	temp1.push_back(i);
	
	sort(temp1.begin(),temp1.end(),cmp1);           //降序排列
    cout<<"down list:"<<endl;
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	
	sort(temp1.begin()+5,temp1.end());               //部分升序
	 cout<<"partly upon list:"<<endl;
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	
	//反转
	reverse(temp1.begin()+1,temp1.end());             //可选择反转范围
	cout<<"after partly reverse:"<<endl;
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
	//查找
	auto iter=find(temp1.begin(),temp1.end()-5,8);              //从begin(),到end()-（不包括它）位置找数字8，找到返回这个位置的指针,找不到就返回设置范围的最后一个指针 
	cout<<*iter;
	
	//copy用法copy(a.begin(),a.end(),b.begin()+1);把a中从a.begin()到a.end()，中的元素从b.begin()+1位置开始复制到b中
	cout<<"temp0's element is :"<<endl;
	for(auto temp:temp0)
	cout<<temp<<ends;
	cout<<endl;
	
	
	cout<<"before copy temp0 to temp1 , temp1's element is :"<<endl;
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	copy(temp0.begin(),temp0.begin()+3,temp1.begin());
	
	cout<<"after copy temp0 to temp1 , temp1's element is :"<<endl;
	for(auto temp:temp1)
	cout<<temp<<ends;
	cout<<endl;
	
/******************************************************************************front()和back()**********************************************/
	cout<<"temp1's last element is "<<temp1.back()<<endl;
	cout<<"temp1's first element is "<<temp1.front()<<endl;
 
}


bool cmp1(double m, double n)
{
     return m>n;	//降序排列
}



```