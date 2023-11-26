#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1843/B

Codeforces Round 881 (Div. 3) B. Long Long 

Today Alex was brought array a_1, a_2, \dots, a_n of length n. He can 
apply as many operations as he wants (including zero operations) to change 
the array elements.

In 1 operation Alex can choose any l and r such that 1 \leq l \leq r \leq 
n, and multiply all elements of the array from l to r inclusive by -1. In 
other words, Alex can replace the subarray [a_l, a_{l + 1}, \dots, a_r] by 
[-a_l, -a_{l + 1}, \dots, -a_r] in 1 operation.

For example, let n = 5, the array is [1, -2, 0, 3, -1], l = 2 and r = 4, 
then after the operation the array will be [1, 2, 0, -3, -1].

Alex is late for school, so you should help him find the maximum possible 
sum of numbers in the array, which can be obtained by making any number of 
operations, as well as the minimum number of operations that must be done 
for this.
*/
void run(){
    int n;scanf("%d",&n);
    ll res=0;int cnt=0;bool pre=false;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        res+=abs(x);
        if(x<=0){
            if(x<0&&pre==false){
                pre=true;cnt++;
            }
        }else pre=false;
    }
    printf("%lld %d\n",res,cnt);
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
