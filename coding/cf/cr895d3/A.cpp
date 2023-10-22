#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1872/problem/0

Codeforces Round 895 (Div. 3) A. Two Vessels 

You have two vessels with water. The first vessel contains a grams of 
water, and the second vessel contains b grams of water. Both vessels are 
very large and can hold any amount of water.

You also have an empty cup that can hold up to c grams of water.

In one move, you can scoop up to c grams of water from any vessel and pour 
it into the other vessel. Note that the mass of water poured in one move 
does not have to be an integer.

What is the minimum number of moves required to make the masses of water 
in the vessels equal? Note that you cannot perform any actions other than 
the described moves.
*/
void run(){
    int a,b,c;scanf("%d%d%d",&a,&b,&c);
    double x=(1.0*a+b)/2;
    double t=abs(1.0*a-x);
    int tt=(t+0.5)/1;
    printf("%d\n",(tt+c-1)/c);
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
