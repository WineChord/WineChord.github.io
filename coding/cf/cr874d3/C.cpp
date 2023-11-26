#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1833/C

Codeforces Round 874 (Div. 3) C. Vlad Building Beautiful Array 

Vlad was given an array a of n positive integers. Now he wants to build a 
beautiful array b of length n from it.

Vlad considers an array beautiful if all the numbers in it are positive 
and have the same parity. That is, all numbers in the beautiful array are 
greater than zero and are either all even or all odd.

To build the array b, Vlad can assign each b_i either the value a_i or a_i 
- a_j, where any j from 1 to n can be chosen.

To avoid trying to do the impossible, Vlad asks you to determine whether 
it is possible to build a beautiful array b of length n using his array a.
*/
#define N 200020
int a[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
    }
    sort(a,a+n);
    bool par=a[0]%2;
    int ev[2]={0};
    ev[a[0]%2]++;
    for(int i=1;i<n;i++){
        if(a[i]%2!=par&&!ev[par!=a[i]%2]){
            puts("NO");
            return;
        }
        ev[a[i]%2]++;
    }
    puts("YES");
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
