#include<bits/stdc++.h>
using namespace std;
using psi=pair<string,int>;
int main(){
    string s;cin>>s;
    unordered_map<string,int> mp;
    string t="atcoder";
    queue<psi> q;q.push({s,0});
    int n=t.size();
    mp[s]=0;
    while(q.size()){
        auto [str,cnt]=q.front();q.pop();
        if(str==t){
            printf("%d\n",cnt);
            return 0;
        }
        for(int i=1;i<n;i++){
            string ss=str;
            swap(ss[i],ss[i-1]);
            if(mp.find(ss)==mp.end())
                q.push({ss,cnt+1});
                mp[ss]=cnt+1;
        }
    }
}