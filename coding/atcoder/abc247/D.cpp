#include<bits/stdc++.h>
#define N 200020
using namespace std;
using ll=long long;
using pii=pair<int,int>;
pii q[N];int hh=0,tt=-1;
int main(){
    int m;scanf("%d",&m);
    while(m--){
        int t;scanf("%d",&t);
        if(t==1){
            int x,c;scanf("%d%d",&x,&c);
            q[++tt]={x,c};
        }else{
            int d;scanf("%d",&d);
            ll res=0;
            while(d){
                auto& [x,c]=q[hh];
                int r=min(c,d);
                d-=r;c-=r;
                res+=1ll*r*x;
                if(c==0)hh++;
            }
            printf("%lld\n",res);
        }
    }
}