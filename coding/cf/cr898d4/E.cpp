#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1873/E

Codeforces Round 898 (Div. 4) E. Building an Aquarium 

You love fish, that's why you have decided to build an aquarium. You have 
a piece of coral made of n columns, the i-th of which is a_i units tall. 
Afterwards, you will build a tank around the coral as follows: 

 Pick an integer h \geq 1 — the height of the tank. Build walls of height 
h on either side of the tank. 

 Then, fill the tank up with water so that the height of each column is h, 
unless the coral is taller than h; then no water should be added to this 
column. 

 For example, with a=[3,1,2,4,6,2,5] and a height of h=4, you will end up 
using a total of w=8 units of water, as shown. <img class="tex-graphics" 
src="https://espresso.codeforces.com/5a69329b8d22a26c9f4de9eb96076db3e8c712
ba.png" style="max-width: 100.0%;max-height: 100.0%;" /> 

 You can use at most x units of water to fill up the tank, but you want to 
build the biggest tank possible. What is the largest value of h you can 
select?
*/
#define N 200020
int a[N];
void run(){
    ll n,x;cin>>n>>x;
    for(int i=0;i<n;i++)cin>>a[i];
    ll l=0,r=1e18;
    auto check=[&](ll m){
        ll res=0;
        for(int i=0;i<n;i++){
            if(a[i]<m)res+=m-a[i];
            if(res>x)return false;
        }
        return true;
    };
    while(l<r){
        ll m=(l+r+1)/2;
        if(check(m))l=m;
        else r=m-1;
    }
    cout<<l<<endl;
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
