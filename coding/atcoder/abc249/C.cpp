#include<bits/stdc++.h>
#define N 22
using namespace std;
string s[N];
int main(){
    int n,k;cin>>n>>k;
    for(int i=0;i<n;i++)cin>>s[i];
    int res=0;
    for(int i=0;i<(1<<n);i++){
        vector<int> sum(26,0);
        for(int j=0;j<n;j++){
            if((i>>j)&1){
                for(auto c:s[j])sum[c-'a']++;
            }
        }
        int cur=0;
        for(int i=0;i<26;i++)cur+=sum[i]==k;
        res=max(res,cur);
    }
    cout<<res<<endl;
}