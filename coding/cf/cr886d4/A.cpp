#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1850/A

Codeforces Round 886 (Div. 4) A. To My Critics 

Suneet has three digits a, b, and c. 

Since math isn't his strongest point, he asks you to determine if you can 
choose any two digits to make a sum greater or equal to 10.

Output "YES" if there is such a pair, and "NO" otherwise.
*/
int a[4];
void run(){
    for(int i=0;i<3;i++)scanf("%d",&a[i]);
    sort(a,a+3);
    if(a[1]+a[2]>=10)puts("YES");
    else puts("NO");
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
