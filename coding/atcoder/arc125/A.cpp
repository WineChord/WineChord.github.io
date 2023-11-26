#include<bits/stdc++.h>
#define N 200020
using namespace std;
int a[N],t[N];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    int d=n;
    for(int i=1;i<n;i++)if(a[i]!=a[0]){
        d=i;break;
    }
    for(int i=n-1;i>0;i--)if(a[i]!=a[0]){
        d=min(d,n-i);break;
    }
    int res=0;bool flag=false;
    int pre=a[0];
    for(int i=0;i<m;i++){
        int x;
        scanf("%d",&x);
        if(pre==x)res++;
        else{
            if(!flag){
                flag=true;
                if(d==n){
                    puts("-1");
                    return 0;
                }
                res+=d;
                res++;
                pre=!pre;
            }else{
                res+=2;
                pre=!pre;
            }
        }
    }
    printf("%d\n",res);
}