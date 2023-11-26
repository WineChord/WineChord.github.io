#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    int n,k;scanf("%d%d",&n,&k);
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        mp[x]++;
    }
    vector<int> a;
    for(auto [k,v]:mp){
        a.push_back(v);
    }
    sort(a.begin(),a.end());
    ll res=0;
    n=a.size();
    for(int i=0;i<n-k;i++){
        res+=a[i];
    }
    printf("%lld\n",res);
}