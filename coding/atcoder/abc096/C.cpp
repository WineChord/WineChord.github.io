#include<bits/stdc++.h>
#define N 55
using namespace std;
char s[N][N];
int dx[4]={0,0,1,-1};
int dy[4]={1,-1,0,0};
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)scanf("%s",s[i]+1);
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++){
            if(s[i][j]!='#')continue;
            bool flag=true;
            for(int k=0;k<4;k++){
                int ni=i+dx[k];
                int nj=j+dy[k];
                if(s[ni][nj]=='#'){
                    flag=false;
                    break;
                }
            }
            if(flag){
                puts("No");
                return 0;
            }
        }
    puts("Yes");
}