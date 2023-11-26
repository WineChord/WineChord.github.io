#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll l,r;cin>>l>>r;
    ll d=r-l;
    while(true){
        for(ll start=l;start+d<=r;start++){
            ll end=start+d;
            if(__gcd(start,end)==1){
                cout<<d<<endl;return 0;
            }
        }
        d--;
    }
}