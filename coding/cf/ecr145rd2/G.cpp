#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1809/problem/G

Educational Codeforces Round 145 (Rated for Div. 2) G. Prediction 

Consider a tournament with n participants. The rating of the i-th 
participant is a_i.

The tournament will be organized as follows. First of all, organizers will 
assign each participant an index from 1 to n. All indices will be unique. 
Let p_i be the participant who gets the index i.

Then, n-1 games will be held. In the first game, participants p_1 and p_2 
will play. In the second game, the winner of the first game will play 
against p_3. In the third game, the winner of the second game will play 
against p_4, and so on — in the last game, the winner of the (n-2)-th game 
will play against p_n.

Monocarp wants to predict the results of all n-1 games (of course, he will 
do the prediction only after the indices of the participants are 
assigned). He knows for sure that, when two participants with ratings x 
and y play, and |x - y| > k, the participant with the higher rating wins. 
But if |x - y| \le k, any of the two participants may win.

Among all n! ways to assign the indices to participants, calculate the 
number of ways to do this so that Monocarp can predict the results of all 
n-1 games. Since the answer can be large, print it modulo 998244353.
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
