#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1891/A

Codeforces Round 907 (Div. 2) A. Sorting with Twos 

You are given an array of integers a_1, a_2, \ldots, a_n. In one 
operation, you do the following: 

 Choose a non-negative integer m, such that 2^m \leq n. 

 Subtract 1 from a_i for all integers i, such that 1 \leq i \leq 2^m. 

Can you sort the array in non-decreasing order by performing some number 
(possibly zero) of operations?

An array is considered non-decreasing if a_i \leq a_{i + 1} for all 
integers i such that 1 \leq i \leq n - 1.
*/
#define N 22
int a[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++){
        scanf("%d",&a[i]);
    }
    for(int p=1;p+1<=n;p*=2){
        for(int i=p+1;i<2*p&&i<n;i++){
            if(a[i+1]<a[i]){
                puts("NO");
                return;
            }
        }
    }
    puts("YES");
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
