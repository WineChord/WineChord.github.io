#include<bits/stdc++.h>
#define N 1000050
using namespace std;
using ll=long long;
char que[N];int hh=0,tt=-1;
int main(){
    int n,q;scanf("%d%d",&n,&q);
    string s;cin>>s;
    ll back=0;
    while(q--){
        int t,x;scanf("%d%d",&t,&x);
        if(t==1){
            back+=x;
            back%=n;
        }else{
            printf("%c\n",s[(n-back+x-1)%n]);
        }
    }
}