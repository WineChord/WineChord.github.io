#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1900/problem/D

Codeforces Round 911 (Div. 2) D. Small GCD 

Let a, b, and c be integers. We define function f(a, b, c) as follows:

Order the numbers a, b, c in such a way that a \le b \le c. Then return 
\gcd(a, b), where \gcd(a, b) denotes the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor (GCD)

 of integers a and b.

So basically, we take the \gcd of the 2 smaller values and ignore the 
biggest one. 

You are given an array a of n elements. Compute the sum of f(a_i, a_j, 
a_k) for each i, j, k, such that 1 \le i &lt; j &lt; k \le n. 

More formally, compute \sum_{i = 1}^n \sum_{j = i+1}^n \sum_{k =j +1}^n 
f(a_i, a_j, a_k). 
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
