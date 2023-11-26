#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1855/A

Codeforces Round 889 (Div. 2) A. Dalton the Teacher 

Dalton is the teacher of a class with n students, numbered from 1 to n. 
The classroom contains n chairs, also numbered from 1 to n. Initially 
student i is seated on chair p_i. It is guaranteed that p_1,p_2,\dots, p_n 
is a permutation of length n.

A student is happy if his/her number is different from the number of 
his/her chair. In order to make all of his students happy, Dalton can 
repeatedly perform the following operation: choose two distinct students 
and swap their chairs. What is the minimum number of moves required to 
make all the students happy? One can show that, under the constraints of 
this problem, it is possible to make all the students happy with a finite 
number of moves.

A permutation of length n is an array consisting of n distinct integers 
from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a permutation, 
but [1,2,2] is not a permutation (2 appears twice in the array), and 
[1,3,4] is also not a permutation (n=3 but there is 4 in the array).
*/
void run(){
    int n;scanf("%d",&n);
    int k=0;
    for(int i=1;i<=n;i++){
        int x;scanf("%d",&x);
        if(x==i)k++;
    }
    printf("%d\n",(k+1)/2);
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
