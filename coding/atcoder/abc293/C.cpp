#include<bits/stdc++.h>
#define N 15
using namespace std;
using ll=long long;
int a[N][N];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++){
            scanf("%d",&a[i][j]);
        }
    vector<int> p;
    for(int i=0;i<n-1;i++)p.push_back(0);
    for(int i=0;i<m-1;i++)p.push_back(1);
    ll res=0;
    do{
        unordered_map<int,int> mp;
        mp[a[0][0]]=1;
        int x=0,y=0;
        for(auto z:p){
            if(z==0)x++;
            else y++;
            mp[a[x][y]]++;
        }
        if(mp.size()==n+m-1)res++;
    }while(next_permutation(p.begin(),p.end()));
    printf("%lld\n",res);
}