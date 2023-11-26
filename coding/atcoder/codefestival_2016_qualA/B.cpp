#include<bits/stdc++.h>
#define N 100010
using namespace std;
int a[N];
int main(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++){
        scanf("%d",&a[i]);
    }
    int res=0;
    for(int i=1;i<=n;i++){
        if(a[a[i]]==i)res++;
    }
    printf("%d\n",res/2);
}