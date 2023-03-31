#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1809/problem/E

Educational Codeforces Round 145 (Rated for Div. 2) E. Two Tanks 

There are two water tanks, the first one fits a liters of water, the 
second one fits b liters of water. The first tank has c (0 \le c \le a) 
liters of water initially, the second tank has d (0 \le d \le b) liters of 
water initially.

You want to perform n operations on them. The i-th operation is specified 
by a single non-zero integer v_i. If v_i > 0, then you try to pour v_i 
liters of water from the first tank into the second one. If v_i &lt; 0, 
you try to pour -v_i liters of water from the second tank to the first one.

When you try to pour x liters of water from the tank that has y liters 
currently available to the tank that can fit z more liters of water, the 
operation only moves \min(x, y, z) liters of water.

For all pairs of the initial volumes of water (c, d) such that 0 \le c \le 
a and 0 \le d \le b, calculate the volume of water in the first tank after 
all operations are performed.
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
