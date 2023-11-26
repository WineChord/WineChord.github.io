#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int dx[8]={-1,0,1,-1,1,-1,0,1};
int dy[8]={-1,-1,-1,0,0,1,1,1};
int main(){
    ll n,m;cin>>n>>m;
    if(n==1&&m==1){
        puts("1");
        return 0;
    }
    if(n==1){
        cout<<max(0ll,m-2)<<endl;
        return 0;
    }
    if(m==1){
        cout<<max(0ll,n-2)<<endl;
        return 0;
    }
    cout<<max(n-2,0ll)*max(m-2,0ll)<<endl;
    // vector<vector<int>> a(n,vector<int>(m,0));
    // for(int i=0;i<n;i++)
    //     for(int j=0;j<m;j++){
    //         a[i][j]^=1;
    //         for(int k=0;k<8;k++){
    //             int ni=i+dx[k];
    //             int nj=j+dy[k];
    //             if(ni<0||nj<0||ni>=n||nj>=m)continue;
    //             a[ni][nj]^=1;
    //         }
    //     }
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<m;j++)
    //         cout<<a[i][j];
    //     cout<<endl;
    // }
}