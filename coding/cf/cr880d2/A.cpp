#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1836/A

Codeforces Round 880 (Div. 2) A. Destroyer 

John is a lead programmer on a destroyer belonging to the space navy of 
the Confederacy of Independent Operating Systems. One of his tasks is 
checking if the electronic brains of robots were damaged during battles.

A standard test is to order the robots to form one or several lines, in 
each line the robots should stand one after another. After that, each 
robot reports the number of robots standing in front of it in its line.

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/1e75235cfc37a2a23f4880cb770ce2a4c546c2
7a.png" style="max-width: 100.0%;max-height: 100.0%;" width="151px" /> An 
example of robots' arrangement (the front of the lines is on the left). 
The robots report the numbers above. 

The i-th robot reported number l_i. Unfortunately, John does not know 
which line each robot stands in, and can't check the reported numbers. 
Please determine if it is possible to form the lines in such a way that 
all reported numbers are correct, or not.
*/
void run(){
    int n;scanf("%d",&n);
    int cnt[110]={0};
    int mx=-1;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        cnt[x]++;
        mx=max(mx,x);
    }
    for(int i=1;i<=mx;i++){
        if(cnt[i]>cnt[i-1]){
            puts("NO");
            return;
        }
    }
    puts("YES");
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
