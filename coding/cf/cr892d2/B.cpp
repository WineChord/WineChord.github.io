#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1859/B

Codeforces Round 892 (Div. 2) B. Olya and Game with Arrays 

Artem suggested a game to the girl Olya. There is a list of n arrays, 
where the i-th array contains m_i \ge 2 positive integers a_{i,1}, 
a_{i,2}, \ldots, a_{i,m_i}.

Olya can move at most one (possibly 0) integer from each array to another 
array. Note that integers can be moved from one array only once, but 
integers can be added to one array multiple times, and all the movements 
are done at the same time.

The beauty of the list of arrays is defined as the sum \sum_{i=1}^n 
\min_{j=1}^{m_i} a_{i,j}. In other words, for each array, we find the 
minimum value in it and then sum up these values.

The goal of the game is to maximize the beauty of the list of arrays. Help 
Olya win this challenging game!
*/
void run(){
    using ll=long long;
    int n;scanf("%d",&n);
    int mi=2e9;vector<ll> mi2;
    for(int i=0;i<n;i++){
        int m;scanf("%d",&m);
        vector<int> a;
        for(int j=0;j<m;j++){
            int x;scanf("%d",&x);
            a.push_back(x);
        }
        sort(a.begin(),a.end());
        mi=min(mi,a[0]);
        mi2.push_back(a[1]);
    }
    sort(mi2.begin(),mi2.end());
    mi2[0]=mi;
    printf("%lld\n",accumulate(mi2.begin(),mi2.end(),0ll));
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
