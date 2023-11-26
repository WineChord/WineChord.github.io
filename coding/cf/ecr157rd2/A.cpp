#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1895/A

Educational Codeforces Round 157 (Rated for Div. 2) A. Treasure Chest 

Monocarp has found a treasure map. The map represents the treasure 
location as an OX axis. Monocarp is at 0, the treasure chest is at x, the 
key to the chest is at y.

Obviously, Monocarp wants to open the chest. He can perform the following 
actions: 

 go 1 to the left or 1 to the right (spending 1 second); 

 pick the key or the chest up if he is in the same point as that object 
(spending 0 seconds); 

 put the chest down in his current point (spending 0 seconds); 

 open the chest if he's in the same point as the chest and has picked the 
key up (spending 0 seconds). 

Monocarp can carry the chest, but the chest is pretty heavy. He knows that 
he can carry it for at most k seconds in total (putting it down and 
picking it back up doesn't reset his stamina).

What's the smallest time required for Monocarp to open the chest?
*/
void run(){
    int x,y,k;scanf("%d%d%d",&x,&y,&k);
    if(y<=x){
        printf("%d\n",x);
        return;
    }
    printf("%d\n",y+max(0,(y-x)-k));
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
