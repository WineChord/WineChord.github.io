#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1870/problem/G

CodeTON Round 6 (Div. 1 + Div. 2, Rated, Prizes!) G. MEXanization 

Let's define f(S). Let S be a multiset (i.e., it can contain repeated 
elements) of non-negative integers. In one operation, you can choose any 
non-empty subset of S (which can also contain repeated elements), remove 
this subset (all elements in it) from S, and add the MEX of the removed 
subset to S. You can perform any number of such operations. After all the 
operations, S should contain exactly 1 number. f(S) is the largest number 
that could remain in S after any sequence of operations.

You are given an array of non-negative integers a of length n. For each of 
its n prefixes, calculate f(S) if S is the corresponding prefix (for the 
i-th prefix, S consists of the first i elements of array a).

The MEX (minimum excluded) of an array is the smallest non-negative 
integer that does not belong to the array. For instance: 

 The MEX of [2,2,1] is 0, because 0 does not belong to the array. 

 The MEX of [3,1,0,1] is 2, because 0 and 1 belong to the array, but 2 
does not. 

 The MEX of [0,3,1,2] is 4, because 0, 1, 2 and 3 belong to the array, but 
4 does not. 
*/
void run(){
    // Welcome, your majesty.
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
