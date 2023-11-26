#include<bits/stdc++.h>
#define N 200020
using namespace std;
using ll=long long;
int a[N];
int main(){
    int n;scanf("%d",&n);
    unordered_map<int,int> mp;
    unordered_map<int,ll> mp1;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        mp[a[i]]++;
    }
    ll sum=0;
    for(auto [k,v]:mp){
        ll vv=1ll*(v-1)*v/2;
        sum+=vv;
        mp1[k]=vv;
    }
    for(int i=0;i<n;i++){
        ll v=mp[a[i]];
        ll vv=(v-2)*(v-1)/2;
        printf("%lld\n",sum-mp1[a[i]]+vv);
    }
}