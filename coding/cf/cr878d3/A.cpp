#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1840/A

Codeforces Round 878 (Div. 3) A. Cipher Shifer 

There is a string a (unknown to you), consisting of lowercase Latin 
letters, encrypted according to the following rule into string s:

 after each character of string a, an arbitrary (possibly zero) number of 
any lowercase Latin letters, different from the character itself, is 
added; 

 after each such addition, the character that we supplemented is added. 

You are given string s, and you need to output the initial string a. In 
other words, you need to decrypt string s.

Note that each string encrypted in this way is decrypted uniquely.
*/
void run(){
    int n;string s;cin>>n>>s;
    string res;
    for(int i=0;i<n;i++){
        char c=s[i];
        res.push_back(c);i++;
        while(i<n&&s[i]!=c)i++;
    }
    cout<<res<<endl;
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
