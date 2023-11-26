#include<bits/stdc++.h>
#define MOD 998244353
using namespace std;
using ll=long long;
int main(){
    int n;cin>>n;
    vector<vector<ll>> f(n+1,vector<ll>(10,0));
    for(int i=1;i<=9;i++)f[1][i]=1;
    ll res=0;
    for(int i=2;i<=n;i++){
        for(int j=1;j<=9;j++){
            for(int k=-1;k<=1;k++){
                int x=j+k;
                if(x<1||x>9)continue;
                f[i][j]=(f[i][j]+f[i-1][x])%MOD;
            }
            if(i==n)res=(res+f[i][j])%MOD;
        }
    }
    cout<<res<<endl;
}