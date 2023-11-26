#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    int n,m;scanf("%d%d",&n,&m);
    vector<int> a;a.push_back(0);
    for(int i=0;i<m;i++){
        int x;scanf("%d",&x);
        a.push_back(x);
    }
    a.push_back(n+1);
    sort(a.begin(),a.end());
    int d=INT_MAX;
    for(int i=1;i<a.size();i++){
        int t=a[i]-a[i-1]-1;
        if(!t)continue;
        d=min(d,a[i]-a[i-1]-1);
    }
    ll res=0;
    for(int i=1;i<a.size();i++){
        int t=a[i]-a[i-1]-1;
        if(!t)continue;
        ll cur=a[i]-a[i-1]-1;
        res+=(cur+d-1)/d;
    }
    printf("%lld\n",res);
}