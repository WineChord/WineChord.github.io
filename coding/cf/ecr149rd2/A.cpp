#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1837/A

Educational Codeforces Round 149 (Rated for Div. 2) A. Grasshopper on a Line 

You are given two integers x and k. Grasshopper starts in a point 0 on an 
OX axis. In one move, it can jump some integer distance, that is not 
divisible by k, to the left or to the right.

What's the smallest number of moves it takes the grasshopper to reach 
point x? What are these moves? If there are multiple answers, print any of 
them.
*/
void run(){
    int x,k;scanf("%d%d",&x,&k);
    int div=x/k;
    int rem=x%k;
    if(rem){
        puts("1");
        printf("%d\n",x);
        return;
    }
    puts("2");
    printf("%d 1\n",div*k-1);
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
