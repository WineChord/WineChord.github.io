#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1846/A

Codeforces Round 883 (Div. 3) A. Rudolph and Cut the Rope  

There are n nails driven into the wall, the i-th nail is driven a_i meters 
above the ground, one end of the b_i meters long rope is tied to it. All 
nails hang at different heights one above the other. One candy is tied to 
all ropes at once. Candy is tied to end of a rope that is not tied to a 
nail.

To take the candy, you need to lower it to the ground. To do this, Rudolph 
can cut some ropes, one at a time. Help Rudolph find the minimum number of 
ropes that must be cut to get the candy.

The figure shows an example of the first test:

  <img class="tex-graphics" 
src="https://espresso.codeforces.com/00f14114dd979e028305fc59f7fa58a0718d91
8f.png" style="max-width: 100.0%;max-height: 100.0%;" width="300px" /> 
*/
void run(){
    int n;scanf("%d",&n);
    int res=0;
    for(int i=0;i<n;i++){
        int x,y;scanf("%d%d",&x,&y);
        if(x>y)res++;
    }
    printf("%d\n",res);
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
