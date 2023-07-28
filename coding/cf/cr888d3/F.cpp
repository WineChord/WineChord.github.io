#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1851/problem/F

Codeforces Round 888 (Div. 3) F. Lisa and the Martians 

Lisa was kidnapped by martians! It okay, because she has watched a lot of 
TV shows about aliens, so she knows what awaits her. Let's call integer 
martian if it is a non-negative integer and strictly less than 2^k, for 
example, when k = 12, the numbers 51, 1960, 0 are martian, and the numbers 
\pi, -1, \frac{21}{8}, 4096 are not.

The aliens will give Lisa n martian numbers a_1, a_2, \ldots, a_n. Then 
they will ask her to name any martian number x. After that, Lisa will 
select a pair of numbers a_i, a_j (i \neq j) in the given sequence and 
count (a_i \oplus x) \&amp; (a_j \oplus x). The operation \oplus means <a 
href="http://tiny.cc/xor_wiki">Bitwise exclusive OR

, the operation \&amp; means <a href="http://tiny.cc/and_wiki ">Bitwise And

. For example, (5 \oplus 17) \&amp; (23 \oplus 17) = (00101_2 \oplus 
10001_2) \&amp; (10111_2 \oplus 10001_2) = 10100_2 \&amp; 00110_2 = 
00100_2 = 4.

Lisa is sure that the higher the calculated value, the higher her chances 
of returning home. Help the girl choose such i, j, x that maximize the 
calculated value.
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
