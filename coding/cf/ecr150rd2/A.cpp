#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1841/A

Educational Codeforces Round 150 (Rated for Div. 2) A. Game with Board 

Alice and Bob play a game. They have a blackboard; initially, there are n 
integers written on it, and each integer is equal to 1.

Alice and Bob take turns; Alice goes first. On their turn, the player has 
to choose several (at least two) equal integers on the board, wipe them 
and write a new integer which is equal to their sum.

For example, if the board currently contains integers \{1, 1, 2, 2, 2, 
3\}, then the following moves are possible:

 choose two integers equal to 1, wipe them and write an integer 2, then 
the board becomes \{2, 2, 2, 2, 3\}; 

 choose two integers equal to 2, wipe them and write an integer 4, then 
the board becomes \{1, 1, 2, 3, 4\}; 

 choose three integers equal to 2, wipe them and write an integer 6, then 
the board becomes \{1, 1, 3, 6\}. 

If a player cannot make a move (all integers on the board are different), 
that player wins the game.

Determine who wins if both players play optimally.
*/
void run(){
    int n;scanf("%d",&n);
    if(n>=5)puts("Alice");
    else puts("Bob");
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
