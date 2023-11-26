#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll d;cin>>d;
    ll mx=sqrt(d);
    ll res=1e12;
    for(ll x=0;x<=mx&&x*x<=d/2;x++){
        ll xx=x*x;
        ll yy=d-xx;
        ll y=sqrt(yy);
        res=min(res,abs(xx+y*y-d));
        res=min(res,abs(xx+(y+1)*(y+1)-d));
    }
    cout<<res<<endl;
}