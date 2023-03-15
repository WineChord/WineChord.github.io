// 9
#include<bits/stdc++.h>
#define MAXN 100010
#define MAXM 200020
using namespace std;
int n,m,h[MAXN],e[MAXM],ne[MAXM],idx,col[MAXN];
void add(int u,int v){
    e[idx]=v;ne[idx]=h[u];h[u]=idx++;
}
bool dfs(int u,int c){
    col[u]=c;
    for(int i=h[u];i!=-1;i=ne[i]){
        int v=e[i];
        if(!col[v]&&!dfs(v,3-c)||col[v]==c)return false;
    }
    return true;
}
int main(){
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);add(v,u);
    }
    bool flag=true;
    for(int i=1;i<=n;i++)
        if(!col[i]&&!dfs(i,1)){
            flag=false;
            break;
        }
    if(flag)puts("Yes");
    else puts("No");
}