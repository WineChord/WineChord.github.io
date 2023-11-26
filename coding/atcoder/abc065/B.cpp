#include<bits/stdc++.h>
#define N 100010
using namespace std;
int a[N];
bool vis[N];
int main(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    int v=1,cur=0;
    while(!vis[v]){
        vis[v]=true;
        if(v==2){
            printf("%d\n",cur);
            return 0;
        }
        v=a[v];cur++;
    }
    puts("-1");
}