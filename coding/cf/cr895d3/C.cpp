#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1872/C

Codeforces Round 895 (Div. 3) C. Non-coprime Split 

You are given two integers l \le r. You need to find positive integers a 
and b such that the following conditions are simultaneously satisfied:

 l \le a + b \le r

 \gcd(a, b) \neq 1

or report that they do not exist.

\gcd(a, b) denotes the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor

 of numbers a and b. For example, \gcd(6, 9) = 3, \gcd(8, 9) = 1, \gcd(4, 
2) = 2.
*/
void run(){
    int l,r;cin>>l>>r;
    if(l==r){
        for(int i=2;i<=l/i;i++)
            if(l%i==0){
                int x=l/i;
                cout<<x<<" "<<x*(i-1)<<endl;
                return;
            }
        puts("-1");
        return;
    }
    if(r<4){
        puts("-1");
        return;
    }
    r-=r%2;
    int x=r/2;
    x-=x%2;
    cout<<x<<" "<<r-x<<endl;
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
