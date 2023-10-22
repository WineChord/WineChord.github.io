#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1882/problem/B

Codeforces Round 899 (Div. 2) B. Sets and Union 

You have n sets of integers S_{1}, S_{2}, \ldots, S_{n}. We call a set S 
attainable, if it is possible to choose some (possibly, none) of the sets 
S_{1}, S_{2}, \ldots, S_{n} so that S is equal to their union^{\dagger}. 
If you choose none of S_{1}, S_{2}, \ldots, S_{n}, their union is an empty 
set.

Find the maximum number of elements in an attainable S such that S \neq 
S_{1} \cup S_{2} \cup \ldots \cup S_{n}.

^{\dagger} The union of sets A_1, A_2, \ldots, A_k is defined as the set 
of elements present in at least one of these sets. It is denoted by A_1 
\cup A_2 \cup \ldots \cup A_k. For example, \{2, 4, 6\} \cup \{2, 3\} \cup 
\{3, 6, 7\} = \{2, 3, 4, 6, 7\}.
*/
void run(){
    int k;scanf("%d",&k);
    vector<vector<int>> a(k);
    unordered_set<int> s;
    for(int i=0;i<k;i++){
        int cnt;scanf("%d",&cnt);
        while(cnt--){
            int x;scanf("%d",&x);
            a[i].push_back(x);
            s.insert(x);
        }
    }
    int res=0;
    for(int i=0;i<(1<<k)-1;i++){
        unordered_set<int> b;
        for(int j=0;j<k;j++){
            if((i>>j)&1)for(auto x:a[j])b.insert(x);
        }
        if(b.size()!=s.size())res=max(res,int(b.size()));
    }
    printf("%d\n",res);
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
