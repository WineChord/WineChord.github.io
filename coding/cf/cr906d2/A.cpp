#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1890/A

Codeforces Round 906 (Div. 2) A. Doremy's Paint 3 

An array b_1, b_2, \ldots, b_n of positive integers is good if all the 
sums of two adjacent elements are equal to the same value. More formally, 
the array is good if there exists a k such that b_1 + b_2 = b_2 + b_3 = 
\ldots = b_{n-1} + b_n = k.

Doremy has an array a of length n. Now Doremy can permute its elements 
(change their order) however she wants. Determine if she can make the 
array good.
*/
#define N 110
int a[N];
void run(){
    int n;scanf("%d",&n);
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        mp[a[i]]++;
    }
    if(mp.size()>2){
        puts("NO");
        return;
    }
    if(mp.size()==1){
        puts("YES");
        return;
    }
    vector<int> res;
    for(auto [k,v]:mp)res.push_back(v);
    if(abs(res[0]-res[1])<=1)puts("YES");
    else puts("NO");
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
