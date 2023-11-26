#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1845/A

Educational Codeforces Round 151 (Rated for Div. 2) A. Forbidden Integer 

You are given an integer n, which you want to obtain. You have an 
unlimited supply of every integer from 1 to k, except integer x (there are 
no integer x at all).

You are allowed to take an arbitrary amount of each of these integers 
(possibly, zero). Can you make the sum of taken integers equal to n?

If there are multiple answers, print any of them.
*/
void run(){
    int n,k,x;scanf("%d%d%d",&n,&k,&x);
    if(x!=1){
        printf("YES\n%d\n",n);
        for(int i=0;i<n;i++)printf("1 ");
        puts("");return;
    }
    if(k==1||(k==2&&n%2)){
        puts("NO");return;
    }
    if(k==2){
        printf("YES\n%d\n",n/2);
        for(int i=0;i<n/2;i++)printf("2 ");
        puts("");return;
    }
    vector<int> res;
    if(n%2)res.push_back(3),n-=3;
    else res.push_back(2),n-=2;
    for(int i=0;i<n/2;i++)res.push_back(2);
    printf("YES\n%d\n",res.size());
    for(auto x:res)printf("%d ",x);
    printf("\n");
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
