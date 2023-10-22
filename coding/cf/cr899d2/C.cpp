#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1882/problem/C

Codeforces Round 899 (Div. 2) C. Card Game 

There are n cards stacked in a deck. Initially, a_{i} is written on the 
i-th card from the top. The value written on a card does not change.

You will play a game. Initially your score is 0. In each turn, you can do 
one of the following operations: 

 Choose an odd^{\dagger} positive integer i, which is not greater than the 
number of cards left in the deck. Remove the i-th card from the top of the 
deck and add the number written on the card to your score. The remaining 
cards will be reindexed starting from the top. 

 Choose an even^{\ddagger} positive integer i, which is not greater than 
the number of cards left in the deck. Remove the i-th card from the top of 
the deck. The remaining cards will be reindexed starting from the top. 

 End the game. You can end the game whenever you want, you do not have to 
remove all cards from the initial deck. 



What is the maximum score you can get when the game ends?

^{\dagger} An integer i is odd, if there exists an integer k such that i = 
2k + 1.

^{\ddagger} An integer i is even, if there exists an integer k such that i 
= 2k.
*/
void run(){
    // Welcome, your majesty.
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
