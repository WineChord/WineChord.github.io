#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1884/A

Codeforces Round 904 (Div. 2) A. Simple Design 

A positive integer is called k-beautiful, if the digit sum of the decimal 
representation of this number is divisible by k^{\dagger}. For example, 
9272 is 5-beautiful, since the digit sum of 9272 is 9 + 2 + 7 + 2 = 20.

You are given two integers x and k. Please find the smallest integer y \ge 
x which is k-beautiful.

^{\dagger} An integer n is divisible by k if there exists an integer m 
such that n = k \cdot m.
*/
bool check(int x,int k){
    int sum=0;
    while(x){
        sum+=x%10;
        x/=10;
    }
    return sum%k==0;
}
void run(){
    int x,k;cin>>x>>k;
    while(!check(x,k))x++;
    printf("%d\n",x);
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
