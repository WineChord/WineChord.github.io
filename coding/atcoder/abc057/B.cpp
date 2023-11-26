#include<bits/stdc++.h>
#define N 55
using namespace std;
int x[N],y[N],c[N],d[N];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++){
        scanf("%d%d",&x[i],&y[i]);
    }
    for(int i=1;i<=m;i++){
        scanf("%d%d",&c[i],&d[i]);
    }
    for(int i=0;i<n;i++){
        int res=1e9;
        int idx=-1;
        for(int j=1;j<=m;j++){
            int dis=abs(x[i]-c[j])+abs(y[i]-d[j]);
            if(dis<res){
                res=dis;idx=j;
            }
        }
        printf("%d\n",idx);
    }
}