#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1857/A

Codeforces Round 891 (Div. 3) A. Array Coloring 

You are given an array consisting of n integers. Your task is to determine 
whether it is possible to color all its elements in two colors in such a 
way that the sums of the elements of both colors have the same parity and 
each color has at least one element colored.

For example, if the array is [1,2,4,3,2,3,5,4], we can color it as 
follows: 
[\color{blue}{1},\color{blue}{2},\color{red}{4},\color{blue}{3},\color{red}
{2},\color{red}{3},\color{red}{5},\color{red}{4}], where the sum of the 
blue elements is 6 and the sum of the red elements is 18.
*/
void run(){
    int n;scanf("%d",&n);
    int sum=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        sum+=x;
    }
    if(sum%2==0)puts("YES");
    else puts("NO");
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
