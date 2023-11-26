#include<bits/stdc++.h>
using namespace std;
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;cin>>T;
    while(T--){
        int n;string s;cin>>n>>s;
        int cnt[2]={0};
        for(auto c:s)cnt[c-'0']++;
        if(cnt[1]%2||cnt[1]==2&&cnt[0]==0||s=="011"||s=="110"){
            puts("-1");
            continue;
        }
        if(cnt[1]!=2){
            cout<<cnt[1]/2<<endl;
            continue;
        }
        if(s=="0110"){
            cout<<3<<endl;
            continue;
        }
        vector<int> a;
        for(int i=0;i<n;i++){
            if(s[i]=='1')a.push_back(i);
        }
        if(a[0]!=a[1]-1){
            cout<<1<<endl;
            continue;
        }
        cout<<2<<endl;
    }
}