#include<bits/stdc++.h>
using namespace std;
using ll=long long;
ll a[40];
ll d[40];
int main(){
    // 0.25 0.5 1 2
    // 1     2  4  8
    cin>>a[1]>>a[2]>>a[4]>>a[8];
    ll n;
    cin>>n;
    n*=4;
    vector<int> b{1,2,4,8};
    sort(b.begin(),b.end(),[&](int x,int y){
        return a[x]*1.0/x<a[y]*1.0/y;
    });
    ll res=0;
    for(auto k:b){
        if(!n)break;
        ll div=n/k;
        ll rem=n%k;
        res+=div*a[k];
        n=rem;
    }
    cout<<res<<endl;
    
}