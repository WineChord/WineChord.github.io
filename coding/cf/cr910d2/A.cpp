#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1898/A

Codeforces Round 910 (Div. 2) A. Milica and String 

Milica has a string s of length n, consisting only of characters A and B. 
She wants to modify s so it contains exactly k instances of B. In one 
operation, she can do the following:

 Select an integer i (1 \leq i \leq n) and a character c (c is equal to 
either A or B). 

 Then, replace each of the first i characters of string s (that is, 
characters s_1, s_2, \ldots, s_i) with c. 

Milica does not want to perform too many operations in order not to waste 
too much time on them.

She asks you to find the minimum number of operations required to modify s 
so it contains exactly k instances of B. She also wants you to find these 
operations (that is, integer i and character c selected in each operation).
*/
void run(){
    int n,k;string s;cin>>n>>k>>s;
    vector<int> pre1(n+1,0);
    vector<int> pre2(n+1,0);
    for(int i=1;i<=n;i++){
        pre1[i]=pre1[i-1]+(s[i-1]=='A');
        pre2[i]=pre2[i-1]+(s[i-1]=='B');
    }
    if(pre2[n]==k){
        puts("0");
        return ;
    }
    if(pre2[n]>k){
        int d=pre2[n]-k;
        for(int i=1;i<=n;i++){
            if(pre2[i]==d){
                puts("1");
                printf("%d A\n",i);
                return;
            }
        }
    }
    int d=k-pre2[n];
    for(int i=1;i<=n;i++){
        if(pre1[i]==d){
            puts("1");
            printf("%d B\n",i);
            return;
        }
    }
    return;
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
