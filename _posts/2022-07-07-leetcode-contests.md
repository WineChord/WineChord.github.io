---
classes: wide2
title: "Leetcode Contests"
excerpt: "Collections of leetcode contests."
categories: 
  - coding
tags: 
  - contests
toc: true
toc_sticky: true
mathjax: true
---
# Weekly Contest 329

## Alternating Digit Sum

```cpp
class Solution {
public:
    int alternateDigitSum(int n) {
        vector<int> a;
        while(n){
            a.push_back(n%10);
            n/=10;
        }
        using ll=long long;
        ll res=0;
        int k=a.size()-1;
        bool flag=true;
        for(int i=k;i>=0;i--){
            if(flag)res+=a[i];
            else res-=a[i];
            flag=!flag;
        }
        return res;
    }
};
```


## Sort the Students by Their Kth Score

```cpp
class Solution {
public:
    vector<vector<int>> sortTheStudents(vector<vector<int>>& a, int k) {
        sort(a.begin(),a.end(),[&](vector<int>& x,vector<int>& y){return x[k]>y[k];});
        return a;
    }
};
```

## Apply Bitwise Operations to Make Strings Equal

```cpp
class Solution {
public:
    bool makeStringsEqual(string s, string t) {
        auto az=[&](string a){
            for(char c:a)if(c=='1')return false;
            return true;
        };
        if(az(s)||az(t))return s==t;
        return true;
    }
};
```

## Minimum Cost to Split an Array

```cpp
class Solution {
public:
    int minCost(vector<int>& a, int K) {
        int n=a.size();
        using ll=long long;
        vector<ll> dp(n+1);
        for(int i=0;i<=n;i++)dp[i]=1e18;
        dp[0]=0;
        int mp[n];
        for(int i=1;i<=n;i++){
            memset(mp,0,sizeof(mp));
            ll cnt=0;
            for(int j=i;j>=1;j--){
                mp[a[j-1]]++;
                if(mp[a[j-1]]==1)cnt++;
                else if(mp[a[j-1]]==2)cnt--;
                dp[i]=min(dp[i],dp[j-1]+i-j+1-cnt+K);
            }
        }
        return dp[n];
    }
};
```


# Weekly Contest 300

## Decode the Message

Easy.

```cpp
class Solution {
public:
    string decodeMessage(string k, string s) {
        unordered_map<char,char> m;
        int i=0;
        for(auto c:k)if(c!=' '&&m.find(c)==m.end()){
            m[c]=(i++)+'a';
        }
        int n=s.length();
        for(int i=0;i<n;i++){
            if(s[i]==' ')continue;
            s[i]=m[s[i]];
        }
        return s;
    }
};
```

```python
class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        mp = {' ': ' '}
        i = 0
        for c in key:
            if c not in mp:
                mp[c] = ascii_lowercase[i]
                i += 1
        return ''.join(mp[c] for c in message)
```

## Spiral Matrix IV

Medium.

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    vector<vector<int>> spiralMatrix(int n, int m, ListNode* head) {
        vector<vector<int>> res(n,vector<int>(m,-1));
        int r=0,c=0,lr=0,lc=0,rr=n,rc=m;
        auto get=[&](){
            int v=head!=nullptr?head->val:-1;
            head=head?head->next:head;
            return v;
        };
        int cnt=0;
        while(head){
            while(head&&c<rc)res[r][c]=get(),c++;c--;r++;
            while(head&&r<rr)res[r][c]=get(),r++;r--;c--;
            while(head&&c>=lc)res[r][c]=get(),c--;c++;r--;
            while(head&&r>lr)res[r][c]=get(),r--;r++;c++;
            lr++;lc++;rr--;rc--;
        }
        return res;
    }
};
```


## Number of People Aware of a Secret

Medium.

Idea: Maintain a queue, loop through each day do the following:
* Pop the front of the queue which has expired.
* Push newly informed people to the end of the queue.

Though there are two loops, in fact, the variable `begin` increases monotonically from `0` to a maximum value `2n`.

The actual time complexity is $O(n)$.

_$n$ is the number of days._

```cpp
class Solution {
    const int MOD=1000000007;
public:
    int peopleAwareOfSecret(int n, int delay, int forget) {
        using ll=long long;
        using pii=pair<ll,int>;
        vector<pii> q; q.push_back(pii{1,1});
        ll res=1;
        int begin=0;
        int i=1+delay;
        for(;i<=n;i++){
            int sz=q.size();ll tot=0;
            for(int j=begin;j<sz;j++){
                auto [num,day]=q[j];
                if(i-day>=forget){
                    res=(res+MOD-num)%MOD;
                    begin++;
                    continue;
                }
                if(i-day>=delay){
                    tot+=num;
                    tot%=MOD;
                    res+=num%MOD;
                    res%=MOD;
                }
            }
            q.push_back({tot,i});
            int nxt=q[begin].second+delay-1;
            if(nxt>i)i=nxt;
        }
        return res; 
    }
};
```

## Number of Increasing Paths in a Grid

Hard.

DFS with memoization. Not very hard.

```cpp
using ll=long long;
#define MAXN 1100
ll mp[1100][1100];
class Solution {
    const int MOD=1000000007;
    int dx[4]={0,0,1,-1};
    int dy[4]={1,-1,0,0};
public:

    int n,m;
    vector<vector<int>> b;
    ll dfs(int x,int y){
        if(mp[x][y]!=0)return mp[x][y];
        ll res=1;
        for(int i=0;i<4;i++){
            int nx=x+dx[i];
            int ny=y+dy[i];
            if(nx<0||ny<0||nx>=n||ny>=m)continue;
            if(b[nx][ny]>b[x][y])res+=dfs(nx,ny);
        }
        return mp[x][y]=res%MOD;
    }
    int countPaths(vector<vector<int>>& a) {
        n=a.size(),m=a[0].size();b=a;
        memset(mp,0,sizeof(ll)*MAXN*MAXN);
        ll res=0;
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                res+=dfs(i,j);
                res%=MOD;
            }
        }
        return res;
    }
};
```
