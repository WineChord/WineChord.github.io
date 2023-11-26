#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1866/A

COMPFEST 15 - Preliminary Online Mirror (Unrated, ICPC Rules, Teams Preferred) A. Ambitious Kid 

Chaneka, Pak Chanek's child, is an ambitious kid, so Pak Chanek gives her 
the following problem to test her ambition.

Given an array of integers [A_1, A_2, A_3, \ldots, A_N]. In one operation, 
Chaneka can choose one element, then increase or decrease the element's 
value by 1. Chaneka can do that operation multiple times, even for 
different elements.

What is the minimum number of operations that must be done to make it such 
that A_1 \times A_2 \times A_3 \times \ldots \times A_N = 0?
*/
void run(){
    int n;scanf("%d",&n);
    int res=0x3f3f3f3f;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        res=min(res,abs(x));
    }
    printf("%d\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
