#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1870/problem/B

CodeTON Round 6 (Div. 1 + Div. 2, Rated, Prizes!) B. Friendly Arrays 

You are given two arrays of integers — a_1, \ldots, a_n of length n, and 
b_1, \ldots, b_m of length m. You can choose any element b_j from array b 
(1 \leq j \leq m), and for all 1 \leq i \leq n perform a_i = a_i | b_j. 
You can perform any number of such operations.

After all the operations, the value of x = a_1 \oplus a_2 \oplus \ldots 
\oplus a_n will be calculated. Find the minimum and maximum values of x 
that could be obtained after performing any set of operations.

Above, | is the <a 
href="https://en.wikipedia.org/wiki/Bitwise_operation#OR">bitwise OR 
operation

, and \oplus is the <a 
href="https://en.wikipedia.org/wiki/Bitwise_operation#XOR">bitwise XOR 
operation

.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    int num=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);num^=x;
    }
    int keep=0;
    for(int i=0;i<m;i++){
        int x;scanf("%d",&x);keep|=x;
    }
    if(n%2){
        printf("%d %d\n",num,num|keep);
    }else{
        printf("%d %d\n",num&(~keep),num);
    }
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
