#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll n,a,b;cin>>n>>a>>b;
    auto fun=[&](ll x){
        ll ed=n/x;
        return (x+x*ed)*ed/2;
    };
    cout<<(1+n)*n/2-fun(a)-fun(b)+fun(a*b/gcd(a,b))<<endl;
}