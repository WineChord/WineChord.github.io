#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1839/A

Codeforces Round 876 (Div. 2) A. The Good Array 

You are given two integers n and k.

An array a_1, a_2, \ldots, a_n of length n, consisting of zeroes and ones 
is good if for all integers i from 1 to n both of the following conditions 
are satisfied:

 at least \lceil \frac{i}{k} \rceil of the first i elements of a are equal 
to 1, 

 at least \lceil \frac{i}{k} \rceil of the last i elements of a are equal 
to 1. 

Here, \lceil \frac{i}{k} \rceil denotes the result of division of i by k, 
rounded up. For example, \lceil \frac{6}{3} \rceil = 2, \lceil 
\frac{11}{5} \rceil = \lceil 2.2 \rceil = 3 and \lceil \frac{7}{4} \rceil 
= \lceil 1.75 \rceil = 2.

Find the minimum possible number of ones in a good array.
*/
void run(){
    int n,k;scanf("%d%d",&n,&k);
    printf("%d\n",(n-1+k-1)/k+1);
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
