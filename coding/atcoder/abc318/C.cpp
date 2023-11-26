#include<bits/stdc++.h>
#define N 200020
using namespace std;
using ll=long long;
ll a[N];
int main(){
    int n,d,p;scanf("%d%d%d",&n,&d,&p);
    for(int i=1;i<=n;i++)scanf("%lld",&a[i]);
    sort(a+1,a+1+n,greater<ll>{});
    for(int i=1;i<=n;i++)a[i]+=a[i-1];
    ll res=0;
    for(int i=1;i<=n;i++){
        int k=min(i+d-1,n);
        if(a[k]-a[i-1]>=p){
            res+=p;
            i=k;
        }else{
            res+=a[n]-a[i-1];
            break;
        }
    }
    printf("%lld\n",res);
}