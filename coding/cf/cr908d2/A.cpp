#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1894/A

Codeforces Round 908 (Div. 2) A. Secret Sport 

Let's consider a game in which two players, A and B, participate. This 
game is characterized by two positive integers, X and Y.

The game consists of sets, and each set consists of plays. In each play, 
exactly one of the players, either A or B, wins. A set ends exactly when 
one of the players reaches X wins in the plays of that set. This player is 
declared the winner of the set. The players play sets until one of them 
reaches Y wins in the sets. After that, the game ends, and this player is 
declared the winner of the entire game.

You have just watched a game but didn't notice who was declared the 
winner. You remember that during the game, n plays were played, and you 
know which player won each play. However, you do not know the values of X 
and Y. Based on the available information, determine who won the entire 
game — A or B. If there is not enough information to determine the winner, 
you should also report it.
*/
void run(){
    int n;scanf("%d",&n);
    string s;cin>>s;
    for(int x=1;x<=n;x++){
        int sa=0,sb=0;
        int wa=0,wb=0;
        int lasta=-1,lastb=-1;
        for(int i=0;i<n;i++){
            if(s[i]=='A')sa++;
            else sb++;
            if(sa>=x||sb>=x){
                if(sa>=x){
                    wa++;lasta=i;
                }else{
                    wb++;lastb=i;
                }
                sa=sb=0;
            }
        }
        if(wa>wb&&lasta==n-1){
            puts("A");return;
        }
        if(wb>wa&&lastb==n-1){
            puts("B");return;
        }
    }
    puts("?");
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
