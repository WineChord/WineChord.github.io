#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1878/B

Codeforces Round 900 (Div. 3) B. Aleksa and Stack 

After the Serbian Informatics Olympiad, Aleksa was very sad, because he 
didn't win a medal (he didn't know stack), so Vasilije came to give him an 
easy problem, just to make his day better.

Vasilije gave Aleksa a positive integer n (n \ge 3) and asked him to 
construct a strictly increasing array of size n of positive integers, such 
that 

 3\cdot a_{i+2} is not divisible by a_i+a_{i+1} for each i (1\le i \le 
n-2). 

 Note that a strictly increasing array a of size n is an array where a_i 
&lt; a_{i+1} for each i (1 \le i \le n-1).Since Aleksa thinks he is a bad 
programmer now, he asked you to help him find such an array.
*/
void run(){
    int n;cin>>n;
    vector<int> a(n,0);
    a[0]=2;a[1]=3;
    for(int i=2;i<n;i++){
        int tot=a[i-2]+a[i-1];
        int can=a[i-1]+1;
        while(can*3%(tot)==0)can++;
        a[i]=can;
    }
    for(auto x:a)printf("%d ",x);
    puts("");
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
