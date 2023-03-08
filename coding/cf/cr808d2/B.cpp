#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1708/B

Codeforces Round 808 (Div. 2) B. Difference of GCDs 

You are given three integers n, l, and r. You need to construct an array 
a_1,a_2,\dots,a_n (l\le a_i\le r) such that \gcd(i,a_i) are all distinct 
or report there's no solution.

Here \gcd(x, y) denotes the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor (GCD)

 of integers x and y.
*/
void run(){
    int n,l,r;scanf("%d%d%d",&n,&l,&r);
    vector<int> res;
    for(int i=1;i<=n;i++){
        int k=((l-1)/i+1)*i;
        if(k>r){
            puts("No\n");
            return;
        }
        res.push_back(k);
    }
    puts("Yes\n");
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
