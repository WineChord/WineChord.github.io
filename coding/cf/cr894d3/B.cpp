#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1862/B

Codeforces Round 894 (Div. 3) B. Sequence Game 

Tema and Vika are playing the following game.

First, Vika comes up with a sequence of positive integers a of length m 
and writes it down on a piece of paper. Then she takes a new piece of 
paper and writes down the sequence b according to the following rule: 

 First, she writes down a_1. 

 Then, she writes down only those a_i (2 \le i \le m) such that a_{i - 1} 
\le a_i. Let the length of this sequence be denoted as n. 

For example, from the sequence a=[4, 3, 2, 6, 3, 3], Vika will obtain the 
sequence b=[4, 6, 3].

She then gives the piece of paper with the sequence b to Tema. He, in 
turn, tries to guess the sequence a.

Tema considers winning in such a game highly unlikely, but still wants to 
find at least one sequence a that could have been originally chosen by 
Vika. Help him and output any such sequence.

Note that the length of the sequence you output should not exceed the 
input sequence length by more than two times.
*/
#define N 200020
int b[N];
void run(){
    int n;scanf("%d",&n);
    int res=1;
    for(int i=0;i<n;i++){
        scanf("%d",&b[i]);
        if(i){
            if(b[i]>=b[i-1])res++;
            else res+=2;
        }
    }
    printf("%d\n",res);
    for(int i=0;i<n;i++){
        printf("%d ",b[i]);
        if(i){
            if(b[i]<b[i-1])printf("%d ",b[i]);
        }
    }
    printf("\n");
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
