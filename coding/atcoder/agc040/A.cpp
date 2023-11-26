#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    string s;cin>>s;
    int n=s.size()+1;
    ll res=0;
    // >>>><<<<>>>>
    // ><
    vector<int> a(n,0);
    int cur=0;
    for(int i=0;i<n-1;i++){
        a[i]=cur;
        if(s[i]=='<')cur++;
        else cur=0;
    }
    a[n-1]=cur;
    // puts("");
    // for(auto c:s)printf(" %c",c);
    // puts("");
    // for(auto x:a)printf("%d ",x);
    // puts("");
    cur=0;
    res=a[n-1];
    for(int i=n-2;i>=0;i--){
        if(s[i]=='>')cur++;
        else cur=0;
        a[i]=max(a[i],cur);
        res+=a[i];
    }
    // puts("");
    // for(auto c:s)printf(" %c",c);
    // puts("");
    // for(auto x:a)printf("%d ",x);
    // puts("");
    printf("%lld\n",res);
}