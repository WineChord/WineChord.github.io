#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1877/A

Codeforces Round 902 (Div. 2, based on COMPFEST 15 - Final Round) A. Goals of Victory 

There are n teams in a football tournament. Each pair of teams match up 
once. After every match, Pak Chanek receives two integers as the result of 
the match, the number of goals the two teams score during the match. The 
efficiency of a team is equal to the total number of goals the team scores 
in each of its matches minus the total number of goals scored by the 
opponent in each of its matches.

After the tournament ends, Pak Dengklek counts the efficiency of every 
team. Turns out that he forgot about the efficiency of one of the teams. 
Given the efficiency of n-1 teams a_1,a_2,a_3,\ldots,a_{n-1}. What is the 
efficiency of the missing team? It can be shown that the efficiency of the 
missing team can be uniquely determined.
*/
void run(){
    int n;cin>>n;n--;
    int res=0;
    while(n--){
        int x;cin>>x;
        res+=x;
    }
    cout<<-res<<endl;
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
