#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    string s;cin>>s;
    int n=s.size();
    vector<int> a(n+1,0);
    ll res=0;
    for(int i=n-1;i>=0;i--){
        a[i]=a[i+1]+(s[i]=='W');
        if(s[i]=='B')res+=a[i];
    }
    printf("%lld\n",res);
}