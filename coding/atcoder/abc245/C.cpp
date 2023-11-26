#include<bits/stdc++.h>
#define N 200020
using namespace std;
int a[N],b[N];
bool f[N][2];
int main(){
    int n,k;scanf("%d%d",&n,&k);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    for(int i=1;i<=n;i++)scanf("%d",&b[i]);
    f[1][0]=f[1][1]=true;
    for(int i=2;i<=n;i++){
        f[i][0]=(f[i-1][0]&&abs(a[i]-a[i-1])<=k)||
                (f[i-1][1]&&abs(a[i]-b[i-1])<=k);
        f[i][1]=(f[i-1][0]&&abs(b[i]-a[i-1])<=k)||
                (f[i-1][1]&&abs(b[i]-b[i-1])<=k);
    }
    if(f[n][0]||f[n][1])puts("Yes");
    else puts("No");
}