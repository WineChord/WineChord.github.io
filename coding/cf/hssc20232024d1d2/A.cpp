#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1864/A

Harbour.Space Scholarship Contest 2023-2024 (Div. 1 + Div. 2) A. Increasing and Decreasing 

You are given three integers x, y, and n.

Your task is to construct an array a consisting of n integers which 
satisfies the following conditions:

 a_1=x, a_n=y; 

 a is strictly increasing (i.e. a_1 &lt; a_2 &lt; \ldots &lt; a_n); 

 if we denote b_i=a_{i+1}-a_{i} for 1 \leq i \leq n-1, then b is strictly 
decreasing (i.e. b_1 > b_2 > \ldots > b_{n-1}). 

If there is no such array a, print a single integer -1.
*/
void run(){
    int x,y,n;scanf("%d%d%d",&x,&y,&n);
    vector<int> a;
    a.push_back(x);
    int tot=(1+n-1)*(n-1)/2;
    if(y-x<tot){
        puts("-1");
        return;
    }
    int start=y-x-tot+n-1;
    a.push_back(x+start);
    int k=n-2;
    for(int i=2;i<n-1;i++){
        a.push_back(a[i-1]+k--);
    }
    for(auto x:a)printf("%d ",x);
    printf("%d\n",y);
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
