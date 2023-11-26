#include<bits/stdc++.h>
#define N 15
using namespace std;
using ll=long long;
ll f[N][2];
int main(){
    int n,x,y;scanf("%d%d%d",&n,&x,&y);
    f[1][0]=1;
    for(int i=2;i<=n;i++){
        f[i][0]=f[i-1][1]+y*f[i-1][0];
        f[i][1]=f[i-1][1]+x*f[i][0];
    }
    printf("%lld\n",f[n][1]);
}