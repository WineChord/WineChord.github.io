#include<bits/stdc++.h>
using namespace std;
using pii=pair<int,int>;
int main(){
    int n,m,h,k;scanf("%d%d%d%d",&n,&m,&h,&k);
    string s;cin>>s;
    map<pii,int> mp;
    for(int i=0;i<m;i++){
        int x,y;scanf("%d%d",&x,&y);
        mp[{x,y}]=1;
    }
    int x=0,y=0;
    for(int i=0;i<n;i++){
        if(s[i]=='R')x++;
        if(s[i]=='L')x--;
        if(s[i]=='U')y++;
        if(s[i]=='D')y--;
        h--;
        if(h<0){
            puts("No");return 0;
        }
        if(h<k&&mp[{x,y}]){
            h=k;
            mp[{x,y}]=0;
        }
    }
    puts("Yes");
}