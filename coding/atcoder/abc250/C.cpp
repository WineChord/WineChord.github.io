#include<bits/stdc++.h>
using namespace std;
int main(){
    int n,q;cin>>n>>q;
    vector<int> a(n);
    iota(a.begin(),a.end(),1);
    unordered_map<int,int> mp;
    for(int i=1;i<=n;i++)mp[i]=i-1;
    while(q--){
        int x;cin>>x;
        int p1=mp[x];
        int p2=p1+1;
        if(p2==n)p2=p1-1;
        swap(a[p1],a[p2]);
        mp[a[p1]]=p1;
        mp[a[p2]]=p2;
    }
    for(int i=0;i<n;i++)cout<<a[i]<<" ";
    cout<<endl;
}