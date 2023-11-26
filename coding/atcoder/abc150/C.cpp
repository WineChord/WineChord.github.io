#include<bits/stdc++.h>
#define N 10
using namespace std;
char p[N],q[N],a[N];
int main(){
    int n;scanf(" %d ",&n);
    for(int i=0;i<n;i++)scanf(" %c ",&p[i]);
    for(int i=0;i<n;i++)scanf(" %c ",&q[i]);
    for(int i=1;i<=n;i++)a[i-1]='0'+i;
    int x=-1,y=-1;
    int cnt=0;
    do{
        if(strcmp(p,a)==0)x=cnt;
        if(strcmp(q,a)==0)y=cnt;
        if(x!=-1&&y!=-1)break;
        cnt++;
    }while(next_permutation(a,a+n));
    printf("%d\n",abs(x-y));
}