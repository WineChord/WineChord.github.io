#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1881/A

Codeforces Round 903 (Div. 3) A. Don't Try to Count 

Given a string x of length n and a string s of length m (n \cdot m \le 
25), consisting of lowercase Latin letters, you can apply any number of 
operations to the string x.

In one operation, you append the current value of x to the end of the 
string x. Note that the value of x will change after this.

For example, if x ="aba", then after applying operations, x will change as 
follows: "aba" \rightarrow "abaaba" \rightarrow "abaabaabaaba".

After what minimum number of operations s will appear in x as a substring? 
A substring of a string is defined as a contiguous segment of it.
*/
void run(){
    int n,m;cin>>n>>m;
    string x,s;cin>>x>>s;
    for(int i=0;i<=5;i++){
        if(x.find(s)!=-1){
            printf("%d\n",i);
            return;
        }
        x+=x;
    }
    puts("-1");
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
