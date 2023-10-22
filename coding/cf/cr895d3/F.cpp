#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1872/problem/F

Codeforces Round 895 (Div. 3) F. Selling a Menagerie 

You are the owner of a menagerie consisting of n animals numbered from 1 
to n. However, maintaining the menagerie is quite expensive, so you have 
decided to sell it!

It is known that each animal is afraid of exactly one other animal. More 
precisely, animal i is afraid of animal a_i (a_i \neq i). Also, the cost 
of each animal is known, for animal i it is equal to c_i.

You will sell all your animals in some fixed order. Formally, you will 
need to choose some permutation^\dagger p_1, p_2, \ldots, p_n, and sell 
animal p_1 first, then animal p_2, and so on, selling animal p_n last.

When you sell animal i, there are two possible outcomes:

 If animal a_i was sold before animal i, you receive c_i money for selling 
animal i.

 If animal a_i was not sold before animal i, you receive 2 \cdot c_i money 
for selling animal i. (Surprisingly, animals that are currently afraid are 
more valuable). 



Your task is to choose the order of selling the animals in order to 
maximize the total profit. 

For example, if a = [3, 4, 4, 1, 3], c = [3, 4, 5, 6, 7], and the 
permutation you choose is [4, 2, 5, 1, 3], then:

 The first animal to be sold is animal 4. Animal a_4 = 1 was not sold 
before, so you receive 2 \cdot c_4 = 12 money for selling it.

 The second animal to be sold is animal 2. Animal a_2 = 4 was sold before, 
so you receive c_2 = 4 money for selling it.

 The third animal to be sold is animal 5. Animal a_5 = 3 was not sold 
before, so you receive 2 \cdot c_5 = 14 money for selling it.

 The fourth animal to be sold is animal 1. Animal a_1 = 3 was not sold 
before, so you receive 2 \cdot c_1 = 6 money for selling it.

 The fifth animal to be sold is animal 3. Animal a_3 = 4 was sold before, 
so you receive c_3 = 5 money for selling it.



Your total profit, with this choice of permutation, is 12 + 4 + 14 + 6 + 5 
= 41. Note that 41 is not the maximum possible profit in this example.

^\dagger A permutation of length n is an array consisting of n distinct 
integers from 1 to n in any order. For example, [2,3,1,5,4] is a 
permutation, but [1,2,2] is not a permutation (2 appears twice in the 
array) and [1,3,4] is also not a permutation (n=3, but 4 is present in the 
array).
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
