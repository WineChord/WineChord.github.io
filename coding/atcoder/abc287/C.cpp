#include<bits/stdc++.h>
#define N 200020
using namespace std;
int in[N];
int fa[N];
int find(int x){
    if(fa[x]==0)return x;
    return fa[x]=find(fa[x]);
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    if(m!=n-1){
        puts("No");
        return 0;
    }
    for(int i=0;i<m;i++){
        int u,v;scanf("%d%d",&u,&v);
        in[u]++;in[v]++;
        int fu=find(u),fv=find(v);
        if(fu!=fv)fa[fu]=fv;
    }
    int cnt1=0,cnt2=0;
    for(int i=1;i<=n;i++){
        if(in[i]==2)cnt2++;
        else if(in[i]==1)cnt1++;
    }
    if(!(cnt1==2&&cnt2==n-2)){
        puts("No");
        return 0;
    }
    unordered_map<int,int> mp;
    for(int i=1;i<=n;i++)mp[find(i)]++;
    if(mp.size()>1)puts("No");
    else puts("Yes");
}