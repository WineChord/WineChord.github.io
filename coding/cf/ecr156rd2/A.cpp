#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1886/A

Educational Codeforces Round 156 (Rated for Div. 2) A. Sum of Three 

Monocarp has an integer n.

He wants to represent his number as a sum of three distinct positive 
integers x, y, and z. Additionally, Monocarp wants none of the numbers x, 
y, and z to be divisible by 3.

Your task is to help Monocarp to find any valid triplet of distinct 
positive integers x, y, and z, or report that such a triplet does not 
exist.
*/
void run(){
    int n;cin>>n;
    if(n<7){
        puts("NO");
        return;
    }
    int r=n%3;
    if(r==1||r==2){
        printf("YES\n%d %d %d\n",1,2,n-3);
        return;
    }
    if(n<=9){
        puts("NO");
        return;
    }
    printf("YES\n%d %d %d\n",1,4,n-5);
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
