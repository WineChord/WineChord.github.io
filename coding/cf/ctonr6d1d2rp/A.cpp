#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1870/problem/A

CodeTON Round 6 (Div. 1 + Div. 2, Rated, Prizes!) A. MEXanized Array 

You are given three non-negative integers n, k, and x. Find the maximum 
possible sum of elements in an array consisting of non-negative integers, 
which has n elements, its MEX is equal to k, and all its elements do not 
exceed x. If such an array does not exist, output -1.

The MEX (minimum excluded) of an array is the smallest non-negative 
integer that does not belong to the array. For instance:

 The MEX of [2,2,1] is 0, because 0 does not belong to the array. 

 The MEX of [3,1,0,1] is 2, because 0 and 1 belong to the array, but 2 
does not. 

 The MEX of [0,3,1,2] is 4, because 0, 1, 2 and 3 belong to the array, but 
4 does not. 
*/
void run(){
    int n,k,x;scanf("%d%d%d",&n,&k,&x);
    if(n<k||x<k-1){
        puts("-1");return;
    }
    int mx=k-1;
    if(x!=k)mx=x;
    printf("%d\n",(0+k-1)*k/2+mx*(n-k));
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
