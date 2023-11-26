#include<bits/stdc++.h>
using namespace std;
int main(){
    using ll=long long;
    int n;scanf("%d",&n);
    ll res=0;
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        res+=x;
        mp[x]++;
    }
    int q;scanf("%d",&q);
    while(q--){
        int b,c;scanf("%d%d",&b,&c);
        res-=mp[b]*1ll*b;
        res+=mp[b]*1ll*c;
        mp[c]+=mp[b];
        mp[b]=0;
        printf("%lld\n",res);
    }
}