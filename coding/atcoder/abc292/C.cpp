#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll n;cin>>n;
    ll res=0;
    for(ll i=1;i<n;i++){
        // ab = i
        // cd = n-i
        ll x=0;
        for(ll j=1;j<=i/j;j++){
            if(i%j)continue;
            x++;
            if(i!=j*j)x++;
        }
        ll y=0;
        for(ll j=1;j<=(n-i)/j;j++){
            if((n-i)%j)continue;
            y++;
            if((n-i)!=j*j)y++;
        }
        res+=x*y;
    }
    cout<<res<<endl;
}