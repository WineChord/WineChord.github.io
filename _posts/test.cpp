// 2
#include<bits/stdc++.h>
#define MAXN 550
#define MAXM 100010
using namespace std;
int n1,n2,m,h[MAXN],e[MAXM],ne[MAXM],idx,match[MAXN],st[MAXN];
void add(int u,int v){
    e[++idx]=v;ne[idx]=h[u];h[u]=idx;
}
bool dfs(int u){
    for(int i=h[u];i;i=ne[i]){
        int v=e[i];
        if(st[v])continue;
        st[v]=1;
        if(!match[v]||dfs(match[v])){
            match[v]=u;
            return true;
        }
    }
    return false;
}
int main(){
    scanf("%d%d%d",&n1,&n2,&m);
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);
    }
    int res=0;
    for(int i=1;i<=n1;i++){
        memset(st,0,sizeof(st));
        if(dfs(i))res++;
    }
    printf("%d\n",res);
}