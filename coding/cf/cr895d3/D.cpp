#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1872/problem/D

Codeforces Round 895 (Div. 3) D. Plus Minus Permutation 

You are given 3 integers — n, x, y. Let's call the score of a 
permutation^\dagger p_1, \ldots, p_n the following value:

(p_{1 \cdot x} + p_{2 \cdot x} + \ldots + p_{\lfloor \frac{n}{x} \rfloor 
\cdot x}) - (p_{1 \cdot y} + p_{2 \cdot y} + \ldots + p_{\lfloor 
\frac{n}{y} \rfloor \cdot y})

In other words, the score of a permutation is the sum of p_i for all 
indices i divisible by x, minus the sum of p_i for all indices i divisible 
by y.

You need to find the maximum possible score among all permutations of 
length n.

For example, if n = 7, x = 2, y = 3, the maximum score is achieved by the 
permutation 
[2,\color{red}{\underline{\color{black}{6}}},\color{blue}{\underline{\color
{black}{1}}},\color{red}{\underline{\color{black}{7}}},5,\color{blue}{\unde
rline{\color{red}{\underline{\color{black}{4}}}}},3] and is equal to (6 + 
7 + 4) - (1 + 4) = 17 - 5 = 12.

^\dagger A permutation of length n is an array consisting of n distinct 
integers from 1 to n in any order. For example, [2,3,1,5,4] is a 
permutation, but [1,2,2] is not a permutation (the number 2 appears twice 
in the array) and [1,3,4] is also not a permutation (n=3, but the array 
contains 4).
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
