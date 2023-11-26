#include<bits/stdc++.h>
#define N 110
using namespace std;
int a[N],b[N],c[N],d[N];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<m;i++){
        scanf("%d%d",&a[i],&b[i]);
    }
    int k;scanf("%d",&k);
    for(int i=0;i<k;i++){
        scanf("%d%d",&c[i],&d[i]);
    }
    int res=0;
    for(int i=0;i<(1<<k);i++){
        int mp[110]={0};
        for(int j=0;j<k;j++){
            if((i>>j)&1)mp[c[j]]=1;
            else mp[d[j]]=1;
        }
        int cnt=0;
        for(int i=0;i<m;i++){
            if(mp[a[i]]&&mp[b[i]])cnt++;
        }
        res=max(res,cnt);
    }
    printf("%d\n",res);
}