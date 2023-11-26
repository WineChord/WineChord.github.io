#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1859/A

Codeforces Round 892 (Div. 2) A. United We Stand 

Given an array a of length n, containing integers. And there are two 
initially empty arrays b and c. You need to add each element of array a to 
exactly one of the arrays b or c, in order to satisfy the following 
conditions:

 Both arrays b and c are non-empty. More formally, let l_b be the length 
of array b, and l_c be the length of array c. Then l_b, l_c \ge 1. 

 For any two indices i and j (1 \le i \le l_b, 1 \le j \le l_c), c_j is 
not a divisor of b_i. 

Output the arrays b and c that can be obtained, or output -1 if they do 
not exist.
*/
#define N 110
int a[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    sort(a,a+n);
    if(a[0]==a[n-1]){
        puts("-1");return;
    }
    int cnt=1;
    while(a[n-cnt-1]==a[n-1])cnt++;
    printf("%d %d\n",n-cnt,cnt);
    for(int i=0;i<n-cnt;i++)printf("%d ",a[i]);
    puts("");
    for(int i=0;i<cnt;i++)printf("%d ",a[n-1]);
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
