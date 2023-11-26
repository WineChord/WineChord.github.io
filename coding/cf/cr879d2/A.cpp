#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1834/A

Codeforces Round 879 (Div. 2) A. Unit Array 

Given an array a of length n, which elements are equal to -1 and 1. Let's 
call the array a good if the following conditions are held at the same 
time:

 a_1 + a_2 + \ldots + a_n \ge 0; 
 a_1 \cdot a_2 \cdot \ldots \cdot a_n = 1. 

In one operation, you can select an arbitrary element of the array a_i and 
change its value to the opposite. In other words, if a_i = -1, you can 
assign the value to a_i := 1, and if a_i = 1, then assign the value to a_i 
:= -1.

Determine the minimum number of operations you need to perform to make the 
array a good. It can be shown that this is always possible.
*/
void run(){
    int n;scanf("%d",&n);
    int a=0,b=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(x>0)a++;
        else b++;
    }
    int res=b%2;
    b-=res;
    a+=res;
    if(b>a){
        int x=(b-a+1)/2;
        if(x%2)x++;
        res+=x;
    }
    printf("%d\n",res);
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
