#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1861/A

Educational Codeforces Round 154 (Rated for Div. 2) A. Prime Deletion 

A prime number is a positive integer that has exactly two different 
positive divisors: 1 and the integer itself. For example, 2, 3, 13 and 101 
are prime numbers; 1, 4, 6 and 42 are not.

You are given a sequence of digits from 1 to 9, in which every digit from 
1 to 9 appears exactly once.

You are allowed to do the following operation several (maybe zero) times: 
choose any digit from the sequence and delete it. However, you cannot 
perform this operation if the sequence consists of only two digits.

Your goal is to obtain a sequence which represents a prime number. Note 
that you cannot reorder the digits in the sequence.

Print the resulting sequence, or report that it is impossible to perform 
the operations so that the resulting sequence is a prime number.
*/
void run(){
    ll x;scanf("%lld",&x);
    while(x){
        int t=x%10;
        if(t==1){
            puts("31");return;
        }else if(t==3){
            puts("13");return;
        }
        x/=10;
    }
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
