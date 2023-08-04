#include<bits/stdc++.h>
using namespace std;
using ll=long long;
void run(){
    ll x,y;scanf("%lld%lld",&x,&y);
    while((x>>y)&1)y++;
    printf("%lld\n",(1LL<<y)-1);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--)run();
}