#include<bits/stdc++.h>
#define N 200020
using namespace std;
using ll=long long;
struct Point{
    ll x,y;
    int idx;
}a[N];
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%lld%lld",&a[i].x,&a[i].y);
        a[i].idx=i+1;
    }
    sort(a,a+n,[&](const Point& x,const Point& y){
        return x.x<y.x;
    });
    vector<Point> can;
    unordered_map<int,int> vis;
    can.push_back(a[0]);
    vis[a[0].idx]=1;
    if(!vis[a[n-1].idx]){
        can.push_back(a[n-1]);
        vis[a[n-1].idx]=1;
    }
    if(!vis[a[1].idx]){
        can.push_back(a[1]);
        vis[a[1].idx]=1;
    }
    if(!vis[a[n-2].idx]){
        can.push_back(a[n-2]);
        vis[a[n-2].idx]=1;
    }
    sort(a,a+n,[&](const Point& x,const Point& y){
        return x.y<y.y;
    });
    if(!vis[a[0].idx]){
        can.push_back(a[0]);
        vis[a[0].idx]=1;
    }
    if(!vis[a[n-1].idx]){
        can.push_back(a[n-1]);
        vis[a[n-1].idx]=1;
    }
    if(!vis[a[1].idx]){
        can.push_back(a[1]);
        vis[a[1].idx]=1;
    }
    if(!vis[a[n-2].idx]){
        can.push_back(a[n-2]);
        vis[a[n-2].idx]=1;
    }
    auto dd=[&](const Point& x,const Point& y){
        return max(abs(x.x-y.x),abs(x.y-y.y));
    };
    vector<ll> dis;
    int sz=can.size();
    for(int i=0;i<sz;i++)
        for(int j=i+1;j<sz;j++){
            auto u=can[i];
            auto v=can[j];
            // printf("# %d %d %d %d\n",u.x,u.y,v.x,v.y);
            dis.push_back(dd(u,v));
        }
    sort(dis.rbegin(),dis.rend());
    printf("%lld\n",dis[1]);
}
