#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1858/A

Codeforces Round 893 (Div. 2) A. Buttons 

Anna and Katie ended up in a secret laboratory.

There are a+b+c buttons in the laboratory. It turned out that a buttons 
can only be pressed by Anna, b buttons can only be pressed by Katie, and c 
buttons can be pressed by either of them. Anna and Katie decided to play a 
game, taking turns pressing these buttons. Anna makes the first turn. Each 
button can be pressed at most once, so at some point, one of the girls 
will not be able to make her turn.

The girl who cannot press a button loses. Determine who will win if both 
girls play optimally.
*/
void run(){
    int a,b,c;scanf("%d%d%d",&a,&b,&c);
    int cnt=(c+1)/2;
    if(a+cnt>b+c-cnt)puts("First");
    else puts("Second");
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
