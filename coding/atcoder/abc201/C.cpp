#include<bits/stdc++.h>
using namespace std;
int main(){
    int res=0;
    string s;cin>>s;
    auto check=[&](int x){
        unordered_map<int,int> mp;
        for(int i=0;i<4;i++){
            mp[x%10]++;x/=10;
        }
        for(int i=0;i<10;i++){
            if(s[i]=='o'&&mp[i]==0)return false;
            if(s[i]=='x'&&mp[i])return false;
        }
        return true;
    };
    for(int i=0;i<10000;i++){
        if(check(i))res++;
    }
    printf("%d\n",res);
}