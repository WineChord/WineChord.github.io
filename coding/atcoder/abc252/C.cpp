#include<bits/stdc++.h>
#define N 110
using namespace std;
string s[N];
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        cin>>s[i];
    }
    int res=1e9;
    for(int i=0;i<10;i++){
        unordered_map<int,int> mp;
        for(int j=0;j<n;j++){
            for(int k=0;k<s[j].size();k++){
                if(s[j][k]==i+'0'){
                    mp[k]++;
                    break;
                }
            }
        }
        int cur=0;
        for(auto [k,v]:mp){
            cur=max(cur,(v-1)*10+k);
        }
        res=min(res,cur);
    }
    printf("%d\n",res);
}