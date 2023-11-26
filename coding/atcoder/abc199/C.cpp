#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;scanf("%d",&n);
    string s;cin>>s;
    int q;scanf("%d",&q);
    bool flip=false;
    while(q--){
        int t,a,b;scanf("%d%d%d",&t,&a,&b);
        if(t==2)flip=!flip;
        else{
            if(!flip)swap(s[a-1],s[b-1]);
            else{
                if(a>n)a-=n;
                else a+=n;
                if(b>n)b-=n;
                else b+=n;
                swap(s[a-1],s[b-1]);
            }
        }
    }
    if(flip){
        for(int i=0;i<n;i++)swap(s[i],s[i+n]);
    }
    cout<<s<<endl;
}