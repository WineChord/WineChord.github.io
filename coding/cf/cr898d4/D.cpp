#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1873/D

Codeforces Round 898 (Div. 4) D. 1D Eraser 

You are given a strip of paper s that is n cells long. Each cell is either 
black or white. In an operation you can take any k consecutive cells and 
make them all white.

Find the minimum number of operations needed to remove all black cells.
*/
void run(){
    int n,k;scanf("%d%d",&n,&k);
    string s;cin>>s;
    int res=0;
    for(int i=0;i<n;i++){
        if(s[i]=='B')res++,i+=k-1;
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
