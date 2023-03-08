#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1304/B

Codeforces Round 620 (Div. 2) B. Longest Palindrome 

Returning back to problem solving, Gildong is now studying about 
palindromes. He learned that a palindrome is a string that is the same as 
its reverse. For example, strings "pop", "noon", "x", and "kkkkkk" are 
palindromes, while strings "moon", "tv", and "abab" are not. An empty 
string is also a palindrome.

Gildong loves this concept so much, so he wants to play with it. He has n 
distinct strings of equal length m. He wants to discard some of the 
strings (possibly none or all) and reorder the remaining strings so that 
the concatenation becomes a palindrome. He also wants the palindrome to be 
as long as possible. Please help him find one.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    string center;
    deque<string> dq;
    unordered_map<string,int> mp;
    for(int i=0;i<n;i++){
        string s;cin>>s;
        mp[s]++;
    }
    for(auto [k,v]:mp){
        int x=v;
        string t=k;
        reverse(t.begin(),t.end());
        if(t!=k){
            int y=mp[t];
            int d=min(x,y);
            mp[k]-=d;
            mp[t]-=d;
            for(int i=0;i<d;i++){
                dq.push_front(k);
                dq.push_back(t);
            }
        }
        if(k==t){
            if(mp[k]>1){
                int d=mp[k]/2;
                mp[k]-=d;
                mp[t]-=d;
                for(int i=0;i<d;i++){
                    dq.push_front(k);
                    dq.push_back(t);
                }
            }
            if(mp[k])center=k,mp[k]--;
            else if(mp[t])center=t,mp[t]--;
        }
    }
    int z=dq.size();
    cout<<z*m+center.size()<<"\n";
    for(int i=0;i<z/2;i++){
        cout<<dq.front();
        dq.pop_front();
    }
    cout<<center;
    for(int i=z/2;i<z;i++){
        cout<<dq.front();
        dq.pop_front();
    }
    cout<<"\n";
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
