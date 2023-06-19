// 1
#include<bits/stdc++.h>
#define N 70
using namespace std;
int a[N],vis[N],n;
int len,tot;
bool dfs(int u,int c,int s){
    if(u*len==tot)return true;
    if(c==len)return dfs(u+1,0,0);
    for(int i=s;i<n;i++){
        if(vis[i]||c+a[i]>len)continue;
        vis[i]=true;
        if(dfs(u,c+a[i],i+1))return true;
        vis[i]=false;
        if(!c||c+a[i]==len)return false;
        int j=i;
        while(j<n&&a[j]==a[i])j++;
        i=j-1;
    }
    return false;
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    while(scanf("%d",&n)!=EOF&&n){
        tot=0;memset(vis,0,sizeof(vis));
        for(int i=0;i<n;i++)scanf("%d",&a[i]),tot+=a[i];
        sort(a,a+n,greater<int>());
        for(len=1;;len++){
            if(tot%len)continue;
            if(dfs(0,0,0)){printf("%d\n",len);break;}
        }
    }
}
