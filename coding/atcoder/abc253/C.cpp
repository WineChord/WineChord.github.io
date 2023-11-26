#include<bits/stdc++.h>
using namespace std;
int main(){
    int q;cin>>q;
    map<int,int> mp;
    while(q--){
        int t,x,c;
        cin>>t;
        if(t==1){
            int x;cin>>x;
            mp[x]++;
        }else if(t==2){
            int x,c;cin>>x>>c;
            mp[x]-=min(mp[x],c);
            if(mp[x]==0)mp.erase(x);
        }else{
            auto it1=mp.begin();
            auto it2=mp.end();
            --it2;
            cout<<it2->first-it1->first<<endl;
        }
    }
}