#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1809/problem/B
 
Educational Codeforces Round 145 (Rated for Div. 2) B. Points on Plane 
 
You are given a two-dimensional plane, and you need to place n chips on 
it. 
 
You can place a chip only at a point with integer coordinates. The cost of 
placing a chip at the point (x, y) is equal to |x| + |y| (where |a| is the 
absolute value of a).
 
The cost of placing n chips is equal to the maximum among the costs of 
each chip.
 
You need to place n chips on the plane in such a way that the distance 
between each pair of chips is strictly greater than 1, and the cost is the 
minimum possible.
*/
void run(){
    ll n;cin>>n;
    ll l=0,r=1e9+10;
    while(l<r){
        ll m=l+(r-l+1)/2;
        ll k=2*m+1;
        if(k*k<=n)l=m;
        else r=m-1;
    }
    ll k=2*l+1;
    if(k*k==n){
        cout<<k-1<<"\n";
        return;
    }
    ll res=k+1;
    l=0,r=1e9+10;
    while(l<r){
        ll m=l+(r-l+1)/2;
        ll k=2*m;
        if(k*k<=n)l=m;
        else r=m-1;
    }
    k=2*l;
    if(k*k==n){
        cout<<k-1<<"\n";
        return;
    }
    cout<<min(res,k+1)<<"\n";
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