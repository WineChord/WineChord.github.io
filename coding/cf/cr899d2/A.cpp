#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1882/problem/A

Codeforces Round 899 (Div. 2) A. Increasing Sequence 

You are given a sequence a_{1}, a_{2}, \ldots, a_{n}. A sequence b_{1}, 
b_{2}, \ldots, b_{n} is called good, if it satisfies all of the following 
conditions: 

 b_{i} is a positive integer for i = 1, 2, \ldots, n; 

 b_{i} \neq a_{i} for i = 1, 2, \ldots, n; 

 b_{1} &lt; b_{2} &lt; \ldots &lt; b_{n}. 

 Find the minimum value of b_{n} among all good sequences b_{1}, b_{2}, 
\ldots, b_{n}.
*/
void run(){
    int n;scanf("%d",&n);
    int k=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(++k==x){
            k++;
        }
    }
    printf("%d\n",k);
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
