#include<bits/stdc++.h>
using namespace std;
int c[3][3];
int a[3],b[3];
int main(){
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            scanf("%d",&c[i][j]);
    for(int i=0;i<3;i++)
        b[i]=c[0][i]-a[0];
    for(int i=1;i<3;i++)
        a[i]=c[i][0]-b[0];
    bool flag=true;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(c[i][j]!=a[i]+b[j]){
                flag=false;
            }
        }
    }
    if(flag)puts("Yes");
    else puts("No");
}