#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll x,k,d;scanf("%lld%lld%lld",&x,&k,&d);
    if(x<0)x=-x;
    if(x/d>=k){
        printf("%lld\n",x-k*d);
        return 0;
    }
    k-=x/d;
    ll res=x%d;
    if(k%2)res=d-res;
    printf("%lld\n",res);
}