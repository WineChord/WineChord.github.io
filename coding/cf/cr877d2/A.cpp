#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1838/A

Codeforces Round 877 (Div. 2) A. Blackboard List 

Two integers were written on a blackboard. After that, the following step 
was carried out n-2 times:

 Select any two integers on the board, and write the absolute value of 
their difference on the board. 

After this process was complete, the list of n integers was shuffled. You 
are given the final list. Recover one of the initial two numbers. You do 
not need to recover the other one.

You are guaranteed that the input can be generated using the above process.
*/
void run(){
    int n;scanf("%d",&n);
    int mi=2e9,mx=-2e9;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        mi=min(mi,x);
        mx=max(mx,x);
    }
    if(mi<0)printf("%d\n",mi);
    else printf("%d\n",mx);
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
