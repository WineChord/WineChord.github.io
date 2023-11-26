#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1890/B

Codeforces Round 906 (Div. 2) B. Qingshan Loves Strings 

Qingshan has a string s, while Daniel has a string t. Both strings only 
contain \texttt{0} and \texttt{1}.

A string a of length k is good if and only if

 a_i \ne a_{i+1} for all i=1,2,\ldots,k-1. 

For example, \texttt{1}, \texttt{101}, \texttt{0101} are good, while 
\texttt{11}, \texttt{1001}, \texttt{001100} are not good.

Qingshan wants to make s good. To do this, she can do the following 
operation any number of times (possibly, zero):

 insert t to any position of s (getting a new s). 

Please tell Qingshan if it is possible to make s good.
*/
bool isp(string& s){
    int n=s.size();
    for(int i=0;i<n-1;i++)
        if(s[i]==s[i+1])return false;
    return true;
}
void run(){
    int n,m;cin>>n>>m;
    string s,t;cin>>s>>t;
    if(isp(s)){
        puts("YES");
        return;
    }
    if(!isp(t)||t[0]!=t[m-1]){
        puts("NO");
        return;
    }
    char c=-1;
    for(int i=0;i<n-1;i++){
        if(s[i]==s[i+1]){
            if(c==-1)c=s[i];
            else if(c!=s[i]){
                puts("NO");
                return;
            }
        }
    }
    if(c==t[0])puts("NO");
    else puts("YES");
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
