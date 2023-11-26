#include<bits/stdc++.h>
#define N 110
using namespace std;
using pii=pair<int,int>;
int a[N],b[N];
bool f[N][10010];
int main(){
    int n,x;scanf("%d%d",&n,&x);
    for(int i=1;i<=n;i++)scanf("%d%d",&a[i],&b[i]);
    f[0][0]=true;
    for(int i=1;i<=n;i++){
        for(int j=0;j<=x;j++){
            f[i][a[i]+j]|=f[i-1][j];
            f[i][b[i]+j]|=f[i-1][j];
        }
    }
    if(f[n][x])puts("Yes");
    else puts("No");
}