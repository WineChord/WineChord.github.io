#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    ll res=1;
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)res*=3;
    ll t=1;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(x%2==0)t*=2;
    }
    printf("%lld\n",res-t);
}