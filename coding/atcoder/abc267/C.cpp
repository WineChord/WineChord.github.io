#include<bits/stdc++.h>
#define N 200020
using namespace std;
using ll=long long;
ll a[N],pre[N];
int main(){
    int n,m;cin>>n>>m;
    for(int i=1;i<=n;i++){
        cin>>a[i];
        pre[i]=pre[i-1]+a[i];
    }
    ll cur=0;
    for(int i=1;i<=m;i++)cur+=i*a[i];
    ll res=cur;
    for(int i=m+1;i<=n;i++){
        cur+=m*a[i];
        cur-=pre[i-1]-pre[i-m-1];
        res=max(res,cur);
    }
    cout<<res<<endl;
}