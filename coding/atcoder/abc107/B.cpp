#include<bits/stdc++.h>
#define N 110
using namespace std;
int s[N][N];
bool kp1[N],kp2[N];
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            scanf(" %c ",&s[i][j]);
    for(int i=0;i<n;i++){
        bool keep=false;
        for(int j=0;j<m;j++){
            if(s[i][j]=='#'){
                keep=true;
                break;
            }
        }
        if(keep)kp1[i]=true;
    }
    for(int i=0;i<m;i++){
        bool keep=false;
        for(int j=0;j<n;j++){
            if(s[j][i]=='#'){
                keep=true;
                break;
            }
        }
        if(keep)kp2[i]=true;
    }
    for(int i=0;i<n;i++){
        if(!kp1[i])continue;
        for(int j=0;j<m;j++){
            if(!kp2[j])continue;
            printf("%c",s[i][j]);
        }
        puts("");
    }
}