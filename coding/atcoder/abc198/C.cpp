#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll r,x,y;scanf("%lld%lld%lld",&r,&x,&y);
    ll rr=r*r;
    ll dd=x*x+y*y;
    if(rr>dd){
        puts("2");return 0;
    }
    if(rr==dd){
        puts("1");return 0;
    }
    double d=sqrt(dd);
    double res=d/r;
    ll ans=ll(res);
    if(res-ans>1e-8)ans++;
    printf("%lld\n",ans);
}