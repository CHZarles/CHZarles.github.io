#include<iostream>
using namespace std;
void qsort(int a[],int l,int r); //l是起点，r是终点
int main()
{
 
    int n;
        cin>>n;
        int list[n];
        for(int i=0;i<n;i++)
        cin>>list[i];
        qsort(list,0,n-1);
        for(int i=0;i<n;i++)
        cout<<list[i]<<' ';
         
 
}
 
/*void qsort(int a[],int l,int r)			//填坑的思路 
{
     if(l>=r)
       return;  //重合,
 
     int i = l; int j = r; int key = a[l];//选择第一个数为key,i左指针，j右指针
 
     while(i<j)             	//还没重合
     {
 
         while(i<j && a[j]>=key)  //从右向左找第一个小于key的值,并且i，j不能相遇 
             j--;					 
              
         //重合或者找到了比key小的数
		      
         if(i<j)                    //先判断有没有重合
         {
             a[i] = a[j];           //填坑 //第一个数已经作为key被保存了，所以开始时可以直接被覆盖 
             i++;                   //及时更新i，避免进不去下面的while
         }
 
         while(i<j && a[i]<key)//从左向右找第一个大于key的值，并且i，j不能相遇。 
             i++;
 
 		//重合或找到了比key大的数
		  
         if(i<j){
             a[j] = a[i];		//填坑 
             j--;
         }
     }
     
     //若是i == j，这一轮的填坑结束，把key填到重合处 
     a[i] = key;
     
     
     qsort(a, l, i-1);//递归调用，排序key值左边的数列  
     
     
     qsort(a, i+1, r);//递归调用，排序key值右边的数列  
	  
 }
 */ 
 void qsort(int a[], int l,int r )
 {
 	if(l>=r) return ;//用>=更健壮 
 	
 	int i=l;int j=r;int key=a[l];
 	
 	while(i<j)
 	{
 		
 		while(i<j&&a[j]>=key)
 		j--;
 		
 		//重合或者找到比key小的 
 		if(i<j)			//如果找到了
		{a[i]=a[j];
		 i++;			//注意这个更新不能放在外面 
		} 
	
		 
		 while(i<j&&a[i]<=key)	//找比key大的
		 i++;
		 
		if(i<j)
		 a[j]=a[i]; 
	
	}
	//i ==j
	a[i]=key;	//填坑
	
/*	写法不清晰 
	mid=(r+l)/2;
	qsort(temp,l,mid);
	qsort(temp,mid+1,right);
*/	
	//直接利用i,j的位置 
	qsort(a,l,i-1);
	qsort(a,i+1,r); 
	  
}
 		
 		
 		
 		
 		
 		
 		
 	
 	
 	 
