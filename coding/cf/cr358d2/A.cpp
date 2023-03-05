#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/682/A

Codeforces Round 358 (Div. 2) A. Alyona and Numbers 

After finishing eating her bun, Alyona came up with two integers n and m. 
She decided to write down two columns of integers — the first column 
containing integers from 1 to n and the second containing integers from 1 
to m. Now the girl wants to count how many pairs of integers she can 
choose, one from the first column and the other from the second column, 
such that their sum is divisible by 5.

Formally, Alyona wants to count the number of pairs of integers (x, y) 
such that 1 ≤ x ≤ n, 1 ≤ y ≤ m and <img align="middle" class="tex-formula" 
src="https://espresso.codeforces.com/172ce45a535a02e9d636dcf4c462a2a1df6c32
15.png" style="max-width: 100.0%;max-height: 100.0%;" /> equals 0.

As usual, Alyona has some troubles and asks you to help.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    int r=n%5;
    int k=n/5;
    ll res=0;
    for(int i=0;i<5;i++){
        if(i==0){
            res+=1ll*k*((i+m)/5);
        }else{
            res+=1ll*((i<=r?1:0)+k)*((i+m)/5);
        }
    }
    printf("%lld\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
