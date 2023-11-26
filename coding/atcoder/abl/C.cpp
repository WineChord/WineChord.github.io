#include<bits/stdc++.h>
#define N 100010
using namespace std;
int fa[N];
int find(int x){
    if(fa[x]==0)return x;
    return fa[x]=find(fa[x]);
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        int pu=find(u);
        int pv=find(v);
        if(pu!=pv){
            fa[pu]=pv;
        }
    }
    unordered_map<int,int> mp;
    for(int i=1;i<=n;i++)mp[find(i)]++;
    printf("%d\n",mp.size()-1);
}