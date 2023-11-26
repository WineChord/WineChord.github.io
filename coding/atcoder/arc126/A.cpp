#include<bits/stdc++.h>
using namespace std;
using ll=long long;
void run(){
    ll n2,n3,n4;scanf("%lld%lld%lld",&n2,&n3,&n4);
    ll n6=n3/2;
    // n2 n4 n6
    // n1 n2 n3
    ll n1=n2;
    n2=n4;
    n3=n6;
    // n1 n2 n3 => 5
    ll d=min(n2,n3);
    ll res=d;
    n2-=d;n3-=d;
    if(n2){
        // n1 n2 
        d=min(n2/2,n1);
        n2-=2*d;n1-=d;
        res+=d;
        if(n2){
            d=min(n2,n1/3);
            n2-=d;n1-=3*d;
            res+=d;
        }
        res+=n1/5;
        printf("%lld\n",res);
        return;
    }
    // n1 n3
    d=min(n3,n1/2);
    n3-=d;n1-=2*d;
    res+=d;
    res+=n1/5;
    printf("%lld\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--)run();
}