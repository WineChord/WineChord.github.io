#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;scanf("%d",&n);
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        mp[x]++;
        mp[x+1]++;
        mp[x-1]++;
    }
    int res=0;
    for(auto [k,v]:mp){
        res=max(res,v);
    }
    printf("%d\n",res);
}