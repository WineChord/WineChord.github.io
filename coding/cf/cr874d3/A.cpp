#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1833/A

Codeforces Round 874 (Div. 3) A. Musical Puzzle 

Vlad decided to compose a melody on his guitar. Let's represent the melody 
as a sequence of notes corresponding to the characters 'a', 'b', 'c', 'd', 
'e', 'f', and 'g'.

However, Vlad is not very experienced in playing the guitar and can only 
record exactly two notes at a time. Vlad wants to obtain the melody s, and 
to do this, he can merge the recorded melodies together. In this case, the 
last sound of the first melody must match the first sound of the second 
melody.

For example, if Vlad recorded the melodies "ab" and "ba", he can merge 
them together and obtain the melody "aba", and then merge the result with 
"ab" to get "abab".

Help Vlad determine the minimum number of melodies consisting of two notes 
that he needs to record in order to obtain the melody s.
*/
void run(){
    int n;string s;cin>>n>>s;
    unordered_set<string> mp;
    for(int i=0;i<n-1;i++){
        mp.insert(s.substr(i,2));
    }
    cout<<mp.size()<<endl;
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
