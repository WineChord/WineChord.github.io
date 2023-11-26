#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1878/A

Codeforces Round 900 (Div. 3) A. How Much Does Daytona Cost? 

We define an integer to be the most common on a subsegment, if its number 
of occurrences on that subsegment is larger than the number of occurrences 
of any other integer in that subsegment. A subsegment of an array is a 
consecutive segment of elements in the array a.

Given an array a of size n, and an integer k, determine if there exists a 
non-empty subsegment of a where k is the most common element.
*/
void run(){
    int n,k;cin>>n>>k;
    bool flag=false;
    for(int i=0;i<n;i++){
        int x;cin>>x;
        if(x==k)flag=true;
    }
    if(flag)puts("YES");
    else puts("NO");
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
