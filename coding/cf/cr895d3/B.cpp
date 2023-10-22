#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1872/problem/B

Codeforces Round 895 (Div. 3) B. The Corridor or There and Back Again 

You are in a corridor that extends infinitely to the right, divided into 
square rooms. You start in room 1, proceed to room k, and then return to 
room 1. You can choose the value of k. Moving to an adjacent room takes 1 
second.

Additionally, there are n traps in the corridor: the i-th trap is located 
in room d_i and will be activated s_i seconds after you enter the room 
\boldsymbol{d_i}. Once a trap is activated, you cannot enter or exit a 
room with that trap.

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/b5c043dc906fc8419a9336f15dbb9f7f1f1b61
1f.png" style="max-width: 100.0%;max-height: 100.0%;" />  A schematic 
representation of a possible corridor and your path to room k and back. 

Determine the maximum value of k that allows you to travel from room 1 to 
room k and then return to room 1 safely.

For instance, if n=1 and d_1=2, s_1=2, you can proceed to room k=2 and 
return safely (the trap will activate at the moment 1+s_1=1+2=3, it can't 
prevent you to return back). But if you attempt to reach room k=3, the 
trap will activate at the moment 1+s_1=1+2=3, preventing your return (you 
would attempt to enter room 2 on your way back at second 3, but the 
activated trap would block you). Any larger value for k is also not 
feasible. Thus, the answer is k=2.
*/
#define INF 0x3f3f3f3f
void run(){
    int n;scanf("%d",&n);
    int res=INF;
    for(int i=1;i<=n;i++){
        int d,s;
        scanf("%d%d",&d,&s);
        res=min(res,d+(s-1)/2);
    }
    printf("%d\n",res);
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
