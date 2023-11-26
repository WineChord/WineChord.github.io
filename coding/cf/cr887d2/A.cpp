#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1853/A

Codeforces Round 887 (Div. 2) A. Desorting 

Call an array a of length n sorted if a_1 \leq a_2 \leq \ldots \leq 
a_{n-1} \leq a_n.

Ntarsis has an array a of length n. 

He is allowed to perform one type of operation on it (zero or more times): 

 Choose an index i (1 \leq i \leq n-1). 

 Add 1 to a_1, a_2, \ldots, a_i. 

 Subtract 1 from a_{i+1}, a_{i+2}, \ldots, a_n. 

The values of a can be negative after an operation.

Determine the minimum operations needed to make a not sorted.
*/
#define N 550
int a[N];
void run(){
    int n;scanf("%d",&n);
    int k=1e9;bool flag=false;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(i)k=min(k,a[i]-a[i-1]);
        if(a[i]<a[i-1])flag=true;
    }
    if(flag)puts("0");
    else printf("%d\n",k/2+1);
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
