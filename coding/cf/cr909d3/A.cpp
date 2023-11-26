#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1899/A

Codeforces Round 909 (Div. 3) A. Game with Integers 

Vanya and Vova are playing a game. Players are given an integer n. On 
their turn, the player can add 1 to the current integer or subtract 1. The 
players take turns; Vanya starts. If after Vanya's move the integer is 
divisible by 3, then he wins. If 10 moves have passed and Vanya has not 
won, then Vova wins.

Write a program that, based on the integer n, determines who will win if 
both players play optimally.
*/
void run(){
    int n;cin>>n;
    if(n%3==0)puts("Second");
    else puts("First");
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
