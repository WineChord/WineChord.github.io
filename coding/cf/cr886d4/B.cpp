#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1850/B

Codeforces Round 886 (Div. 4) B. Ten Words of Wisdom 

In the game show "Ten Words of Wisdom", there are n participants numbered 
from 1 to n, each of whom submits one response. The i-th response is a_i 
words long and has quality b_i. No two responses have the same quality, 
and at least one response has length at most 10.

The winner of the show is the response which has the highest quality out 
of all responses that are not longer than 10 words. Which response is the 
winner?
*/
#define N 55
int a[N],b[N];
void run(){
    int n;scanf("%d",&n);
    int mx=-1,res=-1;
    for(int i=1;i<=n;i++){
        scanf("%d%d",&a[i],&b[i]);
        if(a[i]<=10&&b[i]>mx){
            mx=b[i];
            res=i;
        }
    }
    printf("%d\n",res);
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
