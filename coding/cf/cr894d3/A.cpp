#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1862/A

Codeforces Round 894 (Div. 3) A. Gift Carpet 

Recently, Tema and Vika celebrated Family Day. Their friend Arina gave 
them a carpet, which can be represented as an n \cdot m table of lowercase 
Latin letters.

Vika hasn't seen the gift yet, but Tema knows what kind of carpets she 
likes. Vika will like the carpet if she can read her name on. She reads 
column by column from left to right and chooses one or zero letters from 
current column.

Formally, the girl will like the carpet if it is possible to select four 
distinct columns in order from left to right such that the first column 
contains "v", the second one contains "i", the third one contains "k", and 
the fourth one contains "a".

Help Tema understand in advance whether Vika will like Arina's gift.
*/
#define N 22
char s[N][N];
void run(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)scanf("%s",s[i]);
    string t="vika";
    int state=0;
    for(int i=0;i<m;i++){
        unordered_map<char,int> mp;
        for(int j=0;j<n;j++)
            mp[s[j][i]]++;
        if(mp[t[state]])state++;
        if(state==4){
            puts("YES");return;
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
