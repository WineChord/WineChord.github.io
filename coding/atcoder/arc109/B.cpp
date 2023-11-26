#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll n;cin>>n;
    ll l=1,r=min(n,ll(sqrt((n+1)*2)));
    while(l<r){
        ll m=(l+r+1)/2;
        if((1+m)*m/2<=n+1)l=m;
        else r=m-1;
    }
    cout<<1+n-l<<endl;
}