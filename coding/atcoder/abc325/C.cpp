#include<bits/stdc++.h>
#define N 1010
using namespace std;
char s[N][N];
int n,m;
int vis[N][N];
int dx[8]={-1,-1,-1,0,0,1,1,1};
int dy[8]={1,0,-1,1,-1,1,0,-1};
void dfs(int x,int y){
    vis[x][y]=1;
    for(int i=0;i<8;i++){
        int nx=x+dx[i];
        int ny=y+dy[i];
        if(nx<0||ny<0||nx>=n||ny>=m)continue;
        if(!vis[nx][ny]&&s[nx][ny]=='#')dfs(nx,ny);
    }
}
int main(){
    scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)scanf("%s",s[i]);
    int res=0;
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            if(!vis[i][j]&&s[i][j]=='#'){
                res++;
                dfs(i,j);
            }
    printf("%d\n",res);
}