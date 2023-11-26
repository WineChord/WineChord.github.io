#include<bits/stdc++.h>
#define N 1010
using namespace std;
int p[N],suf[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d",&p[i]);
    suf[n+1]=1e9;
    for(int i=n;i>=1;i--)suf[i]=min(suf[i+1],p[i]);
    int res=0;
    for(int i=1;i<=n;i++)if(p[i]<=suf[i+1])res++;
    printf("%d\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--)run();
}