#include<bits/stdc++.h>
using namespace std;
int main(){
    unordered_map<string,int> mp;
    string t="AKIHABARA";
    queue<string> q;q.push(t);
    while(q.size()){
        auto x=q.front();q.pop();
        mp[x]=1;
        int n=x.size();
        for(int i=0;i<n;i++){
            if(x[i]!='A')continue;
            q.push(x.substr(0,i)+x.substr(i+1,n-i-1));
        }
    }
    string s;cin>>s;
    if(mp[s])puts("YES");
    else puts("NO");
}