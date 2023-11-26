#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1849/A

Educational Codeforces Round 152 (Rated for Div. 2) A. Morning Sandwich 

Monocarp always starts his morning with a good ol' sandwich. Sandwiches 
Monocarp makes always consist of bread, cheese and/or ham.

A sandwich always follows the formula: 

 a piece of bread 

 a slice of cheese or ham 

 a piece of bread 

 \dots 

 a slice of cheese or ham 

 a piece of bread 

So it always has bread on top and at the bottom, and it alternates between 
bread and filling, where filling is a slice of either cheese or ham. Each 
piece of bread and each slice of cheese or ham is called a layer.

Today Monocarp woke up and discovered that he has b pieces of bread, c 
slices of cheese and h slices of ham. What is the maximum number of layers 
his morning sandwich can have?
*/
void run(){
    int b,c,h;scanf("%d%d%d",&b,&c,&h);
    printf("%d\n",min(b-1,c+h)*2+1);
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
