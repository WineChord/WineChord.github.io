#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1842/A

CodeTON Round 5 (Div. 1 + Div. 2, Rated, Prizes!) A. Tenzing and Tsondu 

<div class="epigraph"><div class="epigraph-text">Tsondu always runs first! 
! !

Tsondu and Tenzing are playing a card game. Tsondu has n monsters with 
ability values a_1, a_2, \ldots, a_n while Tenzing has m monsters with 
ability values b_1, b_2, \ldots, b_m.

Tsondu and Tenzing take turns making moves, with Tsondu going first. In 
each move, the current player chooses two monsters: one on their side and 
one on the other side. Then, these monsters will fight each other. Suppose 
the ability values for the chosen monsters are x and y respectively, then 
the ability values of the monsters will become x-y and y-x respectively. 
If the ability value of any monster is smaller than or equal to 0, the 
monster dies.

The game ends when at least one player has no monsters left alive. The 
winner is the player with at least one monster left alive. If both players 
have no monsters left alive, the game ends in a draw.

Find the result of the game when both players play optimally.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    ll a=0,b=0;
    for(int i=0;i<n;i++){
        int x;cin>>x;a+=x;
    }
    for(int i=0;i<m;i++){
        int x;cin>>x;b+=x;
    }
    if(a==b)puts("Draw");
    else if(a>b)puts("Tsondu");
    else puts("Tenzing");
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
