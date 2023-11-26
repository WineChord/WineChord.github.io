#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1900/problem/B

Codeforces Round 911 (Div. 2) B. Laura and Operations 

Laura is a girl who does not like combinatorics. Nemanja will try to 
convince her otherwise.

Nemanja wrote some digits on the board. All of them are either 1, 2, or 3. 
The number of digits 1 is a. The number of digits 2 is b and the number of 
digits 3 is c. He told Laura that in one operation she can do the 
following:

 Select two different digits and erase them from the board. After that, 
write the digit (1, 2, or 3) different from both erased digits. 

For example, let the digits be 1, 1, 1, 2, 3, 3. She can choose digits 1 
and 3 and erase them. Then the board will look like this 1, 1, 2, 3. After 
that, she has to write another digit 2, so at the end of the operation, 
the board will look like 1, 1, 2, 3, 2.

Nemanja asked her whether it was possible for only digits of one type to 
remain written on the board after some operations. If so, which digits can 
they be?

Laura was unable to solve this problem and asked you for help. As an award 
for helping her, she will convince Nemanja to give you some points.
*/
void run(){
    int a[3];
    for(int i=0;i<3;i++)cin>>a[i];
    function<int(int,int,int)> fun=[&](int x,int y,int z){
        if(y==z)return 1;
        if(y>z)swap(y,z);
        // y < z
        int d=z-y;
        if(d%2)return 0;
        x+=y;z-=y;y=0;
        if(x>=d/2)return 1;
        return fun(0,x,z-x);
    };
    cout<<fun(a[0],a[1],a[2])<<" ";
    cout<<fun(a[1],a[0],a[2])<<" ";
    cout<<fun(a[2],a[1],a[0])<<endl;
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
