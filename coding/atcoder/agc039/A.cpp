#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    string s;ll k;cin>>s>>k;
    int n=s.size();
    bool flag=true;
    for(int i=0;i<n-1;i++){
        if(s[i]!=s[i+1]){
            flag=false;
            break;
        }
    }
    if(flag){
        ll tot=1ll*n*k;
        cout<<tot/2<<endl;
        return 0;
    }
    // if(n%2==0){
        if(s[0]!=s[n-1]){
            // aabbaaacc
            ll cnt=0;
            for(int i=0;i<n;i++){
                int j=i;
                while(j<n&&s[j]==s[i])j++;
                ll len=j-i;
                cnt+=len/2;
                i=j-1;
            }
            cout<<cnt*k<<endl;
        }else{
            // abbbaaaa
            ll pre=-1;ll last=0;
            ll cnt=0;
            for(int i=0;i<n;i++){
                int j=i;
                while(j<n&&s[j]==s[i])j++;
                ll len=j-i;
                if(pre==-1)pre=len;
                last=len;
                cnt+=len/2;
                i=j-1;
            }
            cnt-=pre/2+last/2;
            cout<<cnt*k+pre/2+last/2+(pre+last)/2*(k-1)<<endl;
        }
    // }
}