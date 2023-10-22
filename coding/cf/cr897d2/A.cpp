#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1867/problem/A

Codeforces Round 897 (Div. 2) A. green_gold_dog, array and permutation 

green_gold_dog has an array a of length n, and he wants to find a 
permutation b of length n such that the number of distinct numbers in the 
element-wise difference between array a and permutation b is maximized.

A permutation of length n is an array consisting of n distinct integers 
from 1 to n in any order. For example, [2,3,1,5,4] is a permutation, but 
[1,2,2] is not a permutation (as 2 appears twice in the array) and [1,3,4] 
is also not a permutation (as n=3, but 4 appears in the array).

The element-wise difference between two arrays a and b of length n is an 
array c of length n, where c_i = a_i - b_i (1 \leq i \leq n).
*/
void run(){
    int n;scanf("%d",&n);
    using pii=pair<int,int>;
    vector<pii> a;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        a.push_back({x,i});
    }
    vector<int> res(n);
    sort(a.begin(),a.end());
    for(int i=0;i<n;i++){
        auto [v,idx]=a[i];
        res[idx]=n-i;
    }
    for(int i=0;i<n;i++)printf("%d ",res[i]);
    puts("");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--)run();
}
