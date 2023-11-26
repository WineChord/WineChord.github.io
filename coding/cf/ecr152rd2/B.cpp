#include<bits/stdc++.h>
#define N 300030
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1849/B

Educational Codeforces Round 152 (Rated for Div. 2) B. Monsters 

Monocarp is playing yet another computer game. And yet again, his 
character is killing some monsters. There are n monsters, numbered from 1 
to n, and the i-th of them has a_i health points initially.

Monocarp's character has an ability that deals k damage to the monster 
with the highest current health. If there are several of them, the one 
with the smaller index is chosen. If a monster's health becomes less than 
or equal to 0 after Monocarp uses his ability, then it dies.

Monocarp uses his ability until all monsters die. Your task is to 
determine the order in which monsters will die.
*/
int a[N];
void run(){
    int n,k;scanf("%d%d",&n,&k);
    using pii=pair<int,int>;
    priority_queue<pii> q;
    for(int i=1;i<=n;i++){
        scanf("%d",&a[i]);
        a[i]%=k;if(a[i]==0)a[i]=k;
        q.push({a[i],-i});
    }
    vector<int> res;
    while(q.size()){
        auto [h,id]=q.top();q.pop();
        if(h<=k){
            res.push_back(-id);continue;
        }
        q.push({h-k,id});
    }
    for(auto x:res)printf("%d ",x);
    puts("");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
