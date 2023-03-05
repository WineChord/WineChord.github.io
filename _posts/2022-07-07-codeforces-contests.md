---
classes: wide2
title: "Codeforces Contests"
excerpt: "Collections of codeforces contests."
categories: 
  - coding
tags: 
  - contests
toc: true
toc_sticky: true
mathjax: true
---

# Codeforces Round #835 (Div. 4)

[https://codeforces.com/contest/1760](https://codeforces.com/contest/1760)

## A. Medium Number

```cpp
/*
Given three distinct integers a, b, and c, find the medium number between all of them.
*/
#include<bits/stdc++.h>
using namespace std;
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        vector<int> a(3,0);
        for(int i=0;i<3;i++)scanf("%d",&a[i]);
        sort(a.begin(),a.end());
        printf("%d\n",a[1]);
    }
}
```

## B. Atilla's Favorite Problem

```cpp
/*
Atilla needs to write a message which can be represented as a string s. He asks you what is the minimum alphabet size required so that one can write this message.

The alphabet of size x (1≤x≤26) contains only the first x Latin letters. For example an alphabet of size 4 contains only the characters a, b, c and d.
*/
#include<bits/stdc++.h>
using namespace std;
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        int n;
        string s;
        cin>>n>>s;
        int res=1;
        for(int i=0;i<n;i++)if(s[i]-'a'+1>res)res=s[i]-'a'+1;
        printf("%d\n",res);
    }
}
```

## C. Advantage

```cpp
/*
There are n participants in a competition, participant i having a strength of si.

Every participant wonders how much of an advantage they have over the other best participant. In other words, each participant i wants to know the difference between si and sj, where j is the strongest participant in the competition, not counting i (a difference can be negative).

So, they ask you for your help! For each i (1≤i≤n) output the difference between si and the maximum strength of any participant other than participant i.
*/
#include<bits/stdc++.h>
#define MAXN 200020
using namespace std;
using ll=long long;
ll a[MAXN];
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        int n;scanf("%d",&n);
        int mx=0,mm=0;
        scanf("%lld%lld",&a[0],&a[1]);
        if(a[0]>a[1]){
            mx=a[0];mm=a[1];
        }else{
            mx=a[1];mm=a[0];
        }
        for(int i=2;i<n;i++){
            scanf("%lld",&a[i]);
            if(a[i]>mx){
                mm=mx;mx=a[i];
            }else if(a[i]>mm){
                mm=a[i];
            }
        }
        for(int i=0;i<n;i++){
            if(a[i]==mx)printf("%lld",a[i]-mm);
            else printf("%lld",a[i]-mx);
            printf("%c"," \n"[i==n-1]);
        }
    }
}
```

## D. Challenging Valleys

```cpp
/*
You are given an array a[0…n−1] of n integers. This array is called a "valley" if there exists exactly one subarray a[l…r] such that:

0≤l≤r≤n−1,
al=al+1=al+2=⋯=ar,
l=0 or al−1>al,
r=n−1 or ar<ar+1.
*/
#include<bits/stdc++.h>
#define MAXN 200020
using namespace std;
int a[MAXN];
void run(int n){
    bool flag=false;
    for(int i=0;i<n;i++){
        if(i){
            if(a[i]>a[i-1])flag=true;
            else if(flag&&a[i]<a[i-1]){
                printf("NO\n");
                return;
            }
        }
    }
    printf("YES\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        int n;scanf("%d",&n);
        for(int i=0;i<n;i++){
            scanf("%d",&a[i]);
        }
        run(n);
    }
}
```

## E. Binary Inversions

```cpp
/*
You are given a binary array† of length n. You are allowed to perform one operation on it at most once. In an operation, you can choose any element and flip it: turn a 0 into a 1 or vice-versa.

What is the maximum number of inversions‡ the array can have after performing at most one operation?
*/
#include<bits/stdc++.h>
#define MAXN 200020
using namespace std;
using ll=long long;
int a[MAXN],pre1[MAXN],suf0[MAXN];
int first0pos;
int last1pos;
void run(int n){
    pre1[1]=0;
    for(int i=2;i<=n;i++)pre1[i]=pre1[i-1]+(a[i-1]==1);
    suf0[n]=0;
    for(int i=n-1;i>=1;i--)suf0[i]=suf0[i+1]+(a[i+1]==0);
    // int cnt01=-1,cnt10=-1;
    // if(first0pos!=0){
    //     // try 0 => 1
    //     if(pre1[first0pos]<suf0[first0pos]){
    //         cnt01=suf0[first0pos]-pre1[first0pos];
    //     }
    // }
    // if(last1pos!=n+1){
    //     // try 1 => 0
    //     if(suf0[last1pos]<pre1[last1pos]){
    //         cnt10=pre1[last1pos]-suf0[last1pos];
    //     }
    // }
    // if(cnt01>=cnt10&&cnt01>0){
    //     a[first0pos]=1;
    //     for(int i=first0pos+1;i<=n;i++)pre1[i]++;
    //     for(int i=1;i<first0pos;i++)suf0[i]--;
    // }
    // else if(cnt10>0){
    //     a[last1pos]=0;
    //     for(int i=1;i<last1pos;i++)suf0[i]++;
    //     for(int i=last1pos+1;i<=n;i++)pre1[i]--;
    // }
    ll res=0;
    for(int i=1;i<=n;i++)if(a[i])res+=suf0[i];
    ll res0=res;
    if(first0pos!=0){
        res=max(res,res0+suf0[first0pos]-pre1[first0pos]);
    }
    if(last1pos!=n+1){
        res=max(res,res0+pre1[last1pos]-suf0[last1pos]);
    }
    printf("%lld\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        int n;scanf("%d",&n);
        first0pos=0;
        last1pos=n+1;
        for(int i=1;i<=n;i++){
            scanf("%d",&a[i]);
            if(a[i]==0&&first0pos==0)first0pos=i;
        }
        for(int i=n;i>=1;i--){
            if(a[i]==1&&last1pos==n+1){
                last1pos=i;break;
            }
        }
        run(n);
    }
}
```

## F. Quests

```cpp
/*
There are n quests. If you complete the i-th quest, you will gain ai coins. You can only complete at most one quest per day. However, once you complete a quest, you cannot do the same quest again for k days. (For example, if k=2 and you do quest 1 on day 1, then you cannot do it on day 2 or 3, but you can do it again on day 4.)

You are given two integers c and d. Find the maximum value of k such that you can gain at least c coins over d days. If no such k exists, output Impossible. If k can be arbitrarily large, output Infinity.
*/
#include<bits/stdc++.h>
#define MAXN 200020
using namespace std;
using ll=long long;
ll a[MAXN];
ll pre[MAXN];
/*
If `k` can be arbitrarily large, it means the sum of the top `d` of `a_i` is 
greater than or equal to `c`.

If no such `k` exists, it means even if you grab the largest value of `a_i` for 
every day for `d` days, the sum is still less than `c`.

Otherwise, we first sort `a` to descending order, and then calculate the prefix 
sum for the sorted arry. Finally we do binary search on `k`.
*/
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif 
    int T;scanf("%d",&T);
    while(T--){
        ll n,c,d;scanf("%lld%lld%lld",&n,&c,&d);
        for(int i=1;i<=n;i++)scanf("%lld",&a[i]);
        auto cmp=[](ll x,ll y){return x>y;};
        sort(a+1,a+n+1,cmp);
        // printf("after sort: ");
        // for(int i=1;i<=n;i++)printf("%d%c",a[i]," \n"[i==n]);
        if(a[1]*d<c){
            printf("Impossible\n");
            continue;
        }
        for(int i=1;i<=n;i++)pre[i]=pre[i-1]+a[i];
        for(int i=n+1;i<=d;i++)pre[i]=pre[n];
        if(pre[min(d,n)]>=c){
            printf("Infinity\n");
            continue;
        }
        ll l=1,r=d;
        while(l<r){
            ll m=(l+r+1)/2;
            ll cur=pre[m]*(d/m)+pre[d%m];
            if(cur>=c)l=m;
            else r=m-1;
        }
        printf("%lld\n",l-1);
    }
}
```

## G. SlavicG's Favorite Problem

```cpp
/*
You are given a weighted tree with n vertices. Recall that a tree is a connected graph without any cycles. A weighted tree is a tree in which each edge has a certain weight. The tree is undirected, it doesn't have a root.

In a move, you can travel from a node to one of its neighbors (another node it has a direct edge with).

You start with a variable x which is initially equal to 0. When you pass through edge i, x changes its value to x XOR wi (where wi is the weight of the i-th edge).

Your task is to go from vertex a to vertex b, but you are allowed to enter node b if and only if after traveling to it, the value of x will become 0. In other words, you can travel to node b only by using an edge i such that x XOR wi=0. Once you enter node b the game ends and you win.

Additionally, you can teleport at most once at any point in time to any vertex except vertex b. You can teleport from any vertex, even from a.

Answer with "YES" if you can reach vertex b from a, and "NO" otherwise.
*/
#include<bits/stdc++.h>
#define MAXN 100010
using namespace std;
using ll=long long;
struct Edge{
    int v;
    ll w;
};
// if edge[MAXN][MAXN] is used, it will MLE
vector<Edge> edge[MAXN];
int a,b,n;
unordered_set<ll> s; 
void init(){
    s.clear();
    for(int i=0;i<MAXN;i++)edge[i].clear();
}
void addedge(int u,int v,ll w){
    edge[u].push_back({v,w});
}
void dfs1(int u,ll x,int p){
    if(u==b)return;
    s.insert(x);
    for(const auto e:edge[u]){
        int v=e.v;
        ll w=e.w;
        if(v==p)continue;
        dfs1(v,x^w,u);
    }
}
bool dfs2(int u,ll x,int p){
    if(u!=b&&s.count(x))return true;
    for(const auto e:edge[u]){
        int v=e.v;
        ll w=e.w;
        if(v==p)continue;
        if(dfs2(v,x^w,u))return true;
    }
    return false;
}
void run(){
    init();
    scanf("%d%d%d",&n,&a,&b);
    for(int i=1;i<n;i++){
        int u,v;ll w;
        scanf("%d%d%lld",&u,&v,&w);
        addedge(u,v,w);addedge(v,u,w);
    }
    dfs1(a,0,-1); // dfs from a => b
    if(dfs2(b,0,-1))printf("YES\n"); // dfs from b => others
    else printf("NO\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    // unordered_set could be slower than set if
    // the next line is missing (TLE)
    // Reference: https://codeforces.com/blog/entry/22947
    s.max_load_factor(0.25);s.reserve(512);
    while(T--){
        run();
    }
}
```

# Codeforces Round #804 (Div. 2)

## A. The Third Three Number Problem

```cpp
#include<iostream>
#include<cstdio>
using namespace std;
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int t;cin>>t;
    while(t--){
        int n;cin>>n;
        int num=n;
        if(n&1){
            cout<<-1<<"\n";
            continue;
        }
        cout<<n/2<<" "<<"0 0\n";
    }
    return 0;
}
```

## B. Almost Ternary Matrix

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;

void testcase(){
    ll n,m;
    cin>>n>>m;

    for(ll i=0;i<n;i++){
        for(ll j=0;j<m;j++){
            int x=i%4; // 0 1 2 3 
            int y=j%4; // 0 1 2 3 
            if(x==y||x+y==3)cout<<1;
            else cout<<0;
            cout<<" \n"[j==m-1];
        }
    }
}
int main()
{
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    ios_base::sync_with_stdio(false); cin.tie(0);
    int t;
    cin>>t;
    while(t--)
        testcase();
    return 0;
}
```

