#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1858/C

Codeforces Round 893 (Div. 2) C. Yet Another Permutation Problem 

Alex got a new game called "GCD permutations" as a birthday present. Each 
round of this game proceeds as follows:

 First, Alex chooses a permutation^{\dagger} a_1, a_2, \ldots, a_n of 
integers from 1 to n. 

 Then, for each i from 1 to n, an integer d_i = \gcd(a_i, a_{(i \bmod n) + 
1}) is calculated. 

 The score of the round is the number of distinct numbers among d_1, d_2, 
\ldots, d_n. 

Alex has already played several rounds so he decided to find a permutation 
a_1, a_2, \ldots, a_n such that its score is as large as possible.

Recall that \gcd(x, y) denotes the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor (GCD)

 of numbers x and y, and x \bmod y denotes the remainder of dividing x by 
y.

^{\dagger}A permutation of length n is an array consisting of n distinct 
integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a 
permutation, but [1,2,2] is not a permutation (2 appears twice in the 
array), and [1,3,4] is also not a permutation (n=3 but there is 4 in the 
array).
*/
void run(){
    int n;scanf("%d",&n);
    int vis[n+1]={0};
    for(int i=1;i<=n;i++){
        if(vis[i])continue;
        for(int j=i;j<=n;j<<=1){
            vis[j]=1;
            printf("%d ",j);
        }
    }
    printf("\n");
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
