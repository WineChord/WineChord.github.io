#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1873/problem/A

Codeforces Round 898 (Div. 4) A. Short Sort 

There are three cards with letters \texttt{a}, \texttt{b}, \texttt{c} 
placed in a row in some order. You can do the following operation at most 
once: 

 Pick two cards, and swap them. 

 Is it possible that the row becomes \texttt{abc} after the operation? 
Output "YES" if it is possible, and "NO" otherwise.
*/
void run(){
    string s;cin>>s;
    for(int i=0;i<3;i++)
        for(int j=i+1;j<3;j++){
            string t=s;
            swap(t[i],t[j]);
            if(s=="abc"||t=="abc"){
                puts("YES");return;
            }
        }
    puts("NO");return;
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
