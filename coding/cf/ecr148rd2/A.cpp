#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1832/A

Educational Codeforces Round 148 (Rated for Div. 2) A. New Palindrome 

A palindrome is a string that reads the same from left to right as from 
right to left. For example, abacaba, aaaa, abba, racecar are palindromes.

You are given a string s consisting of lowercase Latin letters. The string 
s is a palindrome.

You have to check whether it is possible to rearrange the letters in it to 
get another palindrome (not equal to the given string s).
*/
void run(){
    string s;cin>>s;
    int n=s.size();
    for(int i=1;i<n/2;i++){
        if(s[i]!=s[i-1]){
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
