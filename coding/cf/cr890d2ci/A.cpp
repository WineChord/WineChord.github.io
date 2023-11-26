#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1856/A

Codeforces Round 890 (Div. 2) supported by Constructor Institute A. Tales of a Sort 

Alphen has an array of positive integers a of length n.

Alphen can perform the following operation: 

 For all i from 1 to n, replace a_i with \max(0, a_i - 1). 

Alphen will perform the above operation until a is sorted, that is a 
satisfies a_1 \leq a_2 \leq \ldots \leq a_n. How many operations will 
Alphen perform? Under the constraints of the problem, it can be proven 
that Alphen will perform a finite number of operations.
*/
#define N 55
int a[N],b[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        b[i]=a[i];
    }
    sort(b,b+n);
    int i=n-1;
    while(i>=0){
        if(a[i]!=b[i])break;
        i--;
    }
    if(i==-1)puts("0");
    else printf("%d\n",b[i]);
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
