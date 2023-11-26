#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1900/problem/F

Codeforces Round 911 (Div. 2) F. Local Deletions 

For an array b_1, b_2, \ldots, b_m, for some i (1 &lt; i &lt; m), element 
b_i is said to be a local minimum if b_i &lt; b_{i-1} and b_i &lt; 
b_{i+1}. Element b_1 is said to be a local minimum if b_1 &lt; b_2. 
Element b_m is said to be a local minimum if b_m &lt; b_{m-1}.

For an array b_1, b_2, \ldots, b_m, for some i (1 &lt; i &lt; m), element 
b_i is said to be a local maximum if b_i > b_{i-1} and b_i > b_{i+1}. 
Element b_1 is said to be a local maximum if b_1 > b_2. Element b_m is 
said to be a local maximum if b_m > b_{m-1}.

Let x be an array of distinct elements. We define two operations on it:

 1 — delete all elements from x that are not local minima. 

 2 — delete all elements from x that are not local maxima. 



Define f(x) as follows. Repeat operations 1, 2, 1, 2, \ldots in that order 
until you get only one element left in the array. Return that element.

For example, take an array [1,3,2]. We will first do type 1 operation and 
get [1, 2]. Then we will perform type 2 operation and get [2]. Therefore, 
f([1,3,2]) = 2.

You are given a permutation^\dagger a of size n and q queries. Each query 
consists of two integers l and r such that 1 \le l \le r \le n. The query 
asks you to compute f([a_l, a_{l+1}, \ldots, a_r]). 

^\dagger A permutation of length n is an array of n distinct integers from 
1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, but 
[1,2,2] is not a permutation (2 appears twice in the array), and [1,3,4] 
is also not a permutation (n=3, but there is 4 in the array).
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
