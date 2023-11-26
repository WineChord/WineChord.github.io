#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1844/A

Codeforces Round 884 (Div. 1 + Div. 2) A. Subtraction Game 

You are given two positive integers, a and b (a &lt; b).

For some positive integer n, two players will play a game starting with a 
pile of n stones. They take turns removing exactly a or exactly b stones 
from the pile. The player who is unable to make a move loses.

Find a positive integer n such that the second player to move in this game 
has a winning strategy. This means that no matter what moves the first 
player makes, the second player can carefully choose their moves (possibly 
depending on the first player's moves) to ensure they win.
*/
void run(){
    int a,b;scanf("%d%d",&a,&b);
    printf("%d\n",a+b);
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
