#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1846/B

Codeforces Round 883 (Div. 3) B. Rudolph and Tic-Tac-Toe 

Rudolph invented the game of tic-tac-toe for three players. It has classic 
rules, except for the third player who plays with pluses. Rudolf has a 3 
\times 3 field  — the result of the completed game. Each field cell 
contains either a cross, or a nought, or a plus sign, or nothing. The game 
is won by the player who makes a horizontal, vertical or diagonal row of 
3's of their symbols.

Rudolph wants to find the result of the game. Either exactly one of the 
three players won or it ended in a draw. It is guaranteed that multiple 
players cannot win at the same time.
*/
char s[4][4];
void run(){
    for(int i=0;i<3;i++)scanf("%s",s[i]);
    for(int r=0;r<3;r++){
        unordered_map<char,int> mp;
        for(int c=0;c<3;c++)
            mp[s[r][c]]++;
        auto check=[&](char ch){
            if(mp[ch]==3){
                printf("%c\n",ch);
                return true;
            }
            return false;
        };
        for(auto ch:"OX+")if(check(ch))return;
    }
    for(int c=0;c<3;c++){
        unordered_map<char,int> mp;
        for(int r=0;r<3;r++)
            mp[s[r][c]]++;
        auto check=[&](char ch){
            if(mp[ch]==3){
                printf("%c\n",ch);
                return true;
            }
            return false;
        };
        for(auto ch:"OX+")if(check(ch))return;
    }
    unordered_map<char,int> mp;
    for(int c=0;c<3;c++){
        mp[s[c][c]]++;
    }
    auto check=[&](char ch){
        if(mp[ch]==3){
            printf("%c\n",ch);
            return true;
        }
        return false;
    };
    for(auto ch:"OX+")if(check(ch))return;
    mp.clear();
    for(int c=0;c<3;c++){
        mp[s[2-c][c]]++;
    }
    for(auto ch:"OX+")if(check(ch))return;
    puts("DRAW");
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
