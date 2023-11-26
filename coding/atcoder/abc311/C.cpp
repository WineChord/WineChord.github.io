#include<bits/stdc++.h>
#define N 200020
using namespace std;
int a[N];
bool vis[N];
int main(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++){
        scanf("%d",&a[i]);
    }
    int v=1;
    vector<int> b;
    while(!vis[v]){
        vis[v]=true;
        b.push_back(v);
        v=a[v];
    }
    vector<int> res;res.push_back(v);
    while(b.back()!=v){
        res.push_back(b.back());
        b.pop_back();
    }
    reverse(res.begin(),res.end());
    printf("%d\n",res.size());
    for(auto x:res)printf("%d ",x);
    puts("");
}