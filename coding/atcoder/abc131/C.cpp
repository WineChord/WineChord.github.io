#include<bits/stdc++.h>
using namespace std;
using ll=long long;
using ull=unsigned long long;
int main(){
    ll a,b,c,d;
    cin>>a>>b>>c>>d;
    auto fun=[&](ll x){
        ll st=a/x;
        while(st*x<a){
            st++;
        }
        ll ed=b/x;
        return ed-st+1;
    };
    cout<<(b-a+1)-fun(c)-fun(d)+fun(c*d/gcd(c,d))<<endl;
}