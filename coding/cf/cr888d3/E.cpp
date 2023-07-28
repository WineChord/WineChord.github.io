#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1851/problem/E

Codeforces Round 888 (Div. 3) E. Nastya and Potions 

Alchemist Nastya loves mixing potions. There are a total of n types of 
potions, and one potion of type i can be bought for c_i coins.

Any kind of potions can be obtained in no more than one way, by mixing 
from several others. The potions used in the mixing process will be 
consumed. Moreover, no potion can be obtained from itself through one or 
more mixing processes.

As an experienced alchemist, Nastya has an unlimited supply of k types of 
potions p_1, p_2, \dots, p_k, but she doesn't know which one she wants to 
obtain next. To decide, she asks you to find, for each 1 \le i \le n, the 
minimum number of coins she needs to spend to obtain a potion of type i 
next.
*/
#define N 200020
#define M 500050
struct Edge{
    int to;
}edge[M];
int tot,nxt[M],head[N];
void add(int u,int v){
    edge[++tot]={v};nxt[tot]=head[u];head[u]=tot;
}
int c[N],p[N];
void dfs(int u){
    if(p[u]!=-1)return;
    if(c[u]==0){
        p[u]=0;
        return;
    }
    int res=0;
    for(int i=head[u];i;i=nxt[i]){
        int v=edge[i].to;
        dfs(v);
        res+=p[v];
    }
    p[u]=res;
}
void run(){
    tot=0;memset(c,0,sizeof(c));
    memset(p,-1,sizeof(p));
    int n,k;scanf("%d%d",&n,&k);
    for(int i=1;i<=n;i++){
        scanf("%d",&c[i]);
    }
    for(int i=0;i<k;i++){
        int x;scanf("%d",&x);
        c[x]=0;
    }
    for(int u=1;u<=n;u++){
        int m;scanf("%d",&m);
        while(m--){
            int v;scanf("%d",&v);
            add(u,v);
        }
    }
    for(int i=1;i<=n;i++)dfs(i);
    for(int i=1;i<=n;i++)printf("%d ",p[i]);
    puts("");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
