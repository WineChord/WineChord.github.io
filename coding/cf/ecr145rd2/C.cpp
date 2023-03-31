#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1809/problem/C

Educational Codeforces Round 145 (Rated for Div. 2) C. Sum on Subarrays 

For an array a = [a_1, a_2, \dots, a_n], let's denote its subarray a[l, r] 
as the array [a_l, a_{l+1}, \dots, a_r].

For example, the array a = [1, -3, 1] has 6 non-empty subarrays:

 a[1,1] = [1]; 

 a[1,2] = [1,-3]; 

 a[1,3] = [1,-3,1]; 

 a[2,2] = [-3]; 

 a[2,3] = [-3,1]; 

 a[3,3] = [1]. 

You are given two integers n and k. Construct an array a consisting of n 
integers such that:

 all elements of a are from -1000 to 1000; 

 a has exactly k subarrays with positive sums; 

 the rest \dfrac{(n+1) \cdot n}{2}-k subarrays of a have negative sums. 
*/
void run(){
    int n,k;scanf("%d%d",&n,&k);
    int x=0;
    while((x+1)*(x+2)/2<=k)x++;
    for(int i=0;i<n;i++){
        if(i<x)printf("2");
        else if(i==x)printf("%d",-2*x-1+2*(k-(x*(x+1)/2)));
        else printf("-1000");
        printf("%c"," \n"[i==n-1]);
    }
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
