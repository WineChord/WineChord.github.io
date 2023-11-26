#include<bits/stdc++.h>
#define N 200020
using namespace std;
int fa[N];
int find(int x){
    if(fa[x]==0)return x;
    return fa[x]=find(fa[x]);
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<m;i++){
        int u,v;scanf("%d%d",&u,&v);
        int pu=find(u);
        int pv=find(v);
        if(pu!=pv){
            fa[pu]=pv;
        }
    }
    unordered_set<int> mp;
    for(int i=1;i<=n;i++)mp.insert(find(i));
    int s=mp.size();
    printf("%d\n",m-(n-s));
}