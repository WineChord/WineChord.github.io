#include<bits/stdc++.h>
#define N 10
using namespace std;
int a[N];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    memset(a,-1,sizeof(a));
    while(m--){
        int s,c;
        scanf("%d%d",&s,&c);
        if(s==1&&c==0&&n!=1||a[s]!=-1&&a[s]!=c){
            puts("-1");
            return 0;
        }else a[s]=c;
    }
    if(a[1]==-1){
        if(n!=1)a[1]=1;
        else a[1]=0;
    }
    for(int i=2;i<=n;i++){
        if(a[i]==-1)a[i]=0;
    }
    for(int i=1;i<=n;i++)printf("%d",a[i]);
    puts("");
}