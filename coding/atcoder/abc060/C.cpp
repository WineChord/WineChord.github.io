#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll n,t;cin>>n>>t;
    ll mx=0;
    ll res=0;
    for(int i=0;i<n;i++){
        ll x;cin>>x;
        if(x<mx){
            res-=mx-x;res+=t;mx=x+t;
        }else{
            res+=t;mx=x+t;
        }
    }
    cout<<res<<endl;
}