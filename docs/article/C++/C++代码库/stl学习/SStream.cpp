//科普连接：https://home.gamer.com.tw/creationDetail.php?sn=4114818
#include<iostream>
#include<sstream>
/*strstream类同时可以支持C++风格的串流的输入输出操作。  *\
  是istringstream和ostringstream类的综合，支持<<, >>操作符
\*可以进行字符串到其它类型的快速转换					*/
using namespace std;
int main()
{
	/****************************************************\
	| 重载>>和<<，相当于，以字符串为载体，存储输入输出流 |
	| >>代表寫入stringstream中，<<代表從stringstream拿出 |
	\****************************************************/
	stringstream s1;
    int number =1234;
    string output;//要把number轉成字串型態的容器

    cout<<"number="<<number<<endl;//顯示number=1234;

    s1<<number; //將以int宣告的number放入我們的stringstream中
    s1>>output;

    cout<<"output="<<output<<endl;//顯示output=1234;


    /***********************************\
    |stringstream也可以將string轉成int: |
    \***********************************/
    stringstream string_to_int;
    string s2="12345";
    int n1;

    string_to_int<<s2;
    //也可以使用string_to_int.str(s2);
    //或者 s1=string_to_int.str();

    string_to_int>>n1;

    cout<<"s2="<<s2<<endl;//s2=12345
    cout<<"n1="<<n1<<endl;//n1=12345
    
    /*******************************************************\
	|分割字串的範例:                                        |
	|這邊我們來一個在網路上看到的簡單例題來說明怎麼分割字串 |
	|題目內容:輸入第一行數字N代表接下來有N行資料，接著每行有|
	|不定個數的整數(最多20個，每行最大200個字元)，          |
	|請把每行的總和印出來。                                 |
	|輸入:輸出												|			:
	|4
	|1 1 2 3 5 8 13
	|1 1 5
	|11 557 996 333
	|2 4 6
	|輸出
	|33
	|7
	|1897
	|12
	\********************************************************/
	stringstream s3;
    int N;//代表有幾行
    int i1;//用來存放string轉成int的資料
    while(cin>>N)
    {
      cin.ignore();//估计是跳过换行 :https://www.cnblogs.com/ranjiewen/p/5582601.html
      string line;//讀入每行的資料
      for(int i=0;i<N;i++)
      {
         getline(cin,line);//讀入每行的資料
         int sum=0;//計算總和
         s3.clear();//清除緩存
         s3<<line;
         //也可以使用s3.str(line);
         //還可以寫成line=s3.str();
         while(true)
         {
           s3>>i1;
           if(s3.fail()) break;//確認stringstream有正常流出，沒有代表空了
           sum+=i1;//把每個轉換成int的數字都丟入sum累加
         }
         cout<<sum<<endl;
      }
    }
    
    /***********************************************\
    針對stringstream類別的初始化                    |
	這邊要提到一點就是要重複使用一個stringstream的  |
	情況，因為宣告stringstream類別的時候其實蠻消耗  |
	CPU的時間，在解題目以及應用的時候不太建議重複宣 |
	告，而是使用完之後就初始化再次利用。            |
	基本就是要以下這兩行:                           |
	\***********************************************/
	stringstream s4;
    s4.str("");
    s4.clear();
	
}
