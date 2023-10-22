#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1861/B

Educational Codeforces Round 154 (Rated for Div. 2) B. Two Binary Strings 

You are given two strings a and b of equal length, consisting of only 
characters 0 and/or 1; both strings start with character 0 and end with 
character 1. 

You can perform the following operation any number of times (possibly 
zero): 

 choose one of the strings and two equal characters in it; then turn all 
characters between them into those characters. 

Formally, you choose one of these two strings (let the chosen string be 
s), then pick two integers l and r such that 1 \le l &lt; r \le |s| and 
s_l = s_r, then replace every character s_i such that l &lt; i &lt; r with 
s_l.

For example, if the chosen string is 010101, you can transform it into one 
of the following strings by applying one operation:

 000101 if you choose l = 1 and r = 3; 

 000001 if you choose l = 1 and r = 5; 

 010001 if you choose l = 3 and r = 5; 

 010111 if you choose l = 4 and r = 6; 

 011111 if you choose l = 2 and r = 6; 

 011101 if you choose l = 2 and r = 4. 

You have to determine if it's possible to make the given strings equal by 
applying this operation any number of times.
*/
void run(){
    string a,b;cin>>a>>b;
    int n=a.size();
    for(int i=0;i<n-1;i++){
        if(a[i]=='0'&&a[i]==b[i]&&a[i+1]=='1'&&a[i+1]==b[i+1]){
            puts("YES");
            return;
        }
    }
    puts("NO");
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
