#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1847/A

Codeforces Round 882 (Div. 2) A. The Man who became a God  

Kars is tired and resentful of the narrow mindset of his village since 
they are content with staying where they are and are not trying to become 
the perfect life form. Being a top-notch inventor, Kars wishes to enhance 
his body and become the perfect life form. Unfortunately, n of the 
villagers have become suspicious of his ideas. The i-th villager has a 
suspicion of a_i on him. Individually each villager is scared of Kars, so 
they form into groups to be more powerful.

The power of the group of villagers from l to r be defined as f(l,r) where 

f(l,r) = |a_l - a_{l+1}| + |a_{l + 1} - a_{l + 2}| + \ldots + |a_{r-1} - 
a_r|.

Here |x-y| is the absolute value of x-y. A group with only one villager 
has a power of 0.

Kars wants to break the villagers into exactly k contiguous subgroups so 
that the sum of their power is minimized. Formally, he must find k - 1 
positive integers 1 \le r_1 &lt; r_2 &lt; \ldots &lt; r_{k - 1} &lt; n 
such that f(1, r_1) + f(r_1 + 1, r_2) + \ldots + f(r_{k-1} + 1, n) is 
minimised. Help Kars in finding the minimum value of f(1, r_1) + f(r_1 + 
1, r_2) + \ldots + f(r_{k-1} + 1, n).
*/
#define N 550
int a[N];
void run(){
    int n,k;scanf("%d%d",&n,&k);
    vector<int> d;
    int res=0;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(i){
            d.push_back(abs(a[i]-a[i-1]));
            res+=d.back();
        }
    }
    sort(d.rbegin(),d.rend());
    for(int i=0;i<k-1;i++){
        res-=d[i];
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
