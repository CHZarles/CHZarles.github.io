```C++
#include <iostream>
#include <string>
#include <stack>
#include <queue>
using namespace std;
/*
	（1）构造：std::atack<int >myStack;
	（2）通用操作：empty(), size(),
	（3）特有操作：pop(), push(),top(),
*/
void  mystackTest()
{	
	cout << "---------------------------this is class stack demo--------------------------------------" << endl;
	stack<int> mystack;
	 mystack.push(1);
	 mystack.push(2);
	 mystack.push(0);									//元素进栈
	 
    int top_num=mystack.top();
    cout<<"the num of mystack is: "<<top_num<<endl;     //栈顶元素
    mystack.pop();                                      //弹出栈顶元素
    int top_num1=mystack.top();
    cout<<"the num of mystack is: "<<top_num1<<endl;
    
    return ;
}

/*
	（1）构造：  std::queue<int> myQueue;
	（2）通用操作：empty(), size(), front(), back(),
	（3）特有操作：push(), pop(),
*/

void myquequeTest()
{
	cout << "---------------------------this is class queue demo--------------------------" << std::endl;
	queue<int>myqueue;
	myqueue.push(1);
	myqueue.push(2);
	myqueue.push(3);                                    //入队列
	int front_num=myqueue.front();                      //队头元素
	int back_num=myqueue.back();                        //队尾元素
	printf("the front and back num of the myQueue : %d and %d \n", front_num, back_num);

    myqueue.pop();                                      //弹出元素不返回元素
    myqueue.pop();
    int front_num1=myqueue.front();
    int back_num1=myqueue.back();
    printf("the front and back num of the myQueue : %d and %d \n", front_num1, back_num1);
	
}

int main()
{
	 myquequeTest();
	 mystackTest();
}

```