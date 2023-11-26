#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1900/problem/A

Codeforces Round 911 (Div. 2) A. Cover in Water 

Filip has a row of cells, some of which are blocked, and some are empty. 
He wants all empty cells to have water in them. He has two actions at his 
disposal:

 1 — place water in an empty cell. 

 2 — remove water from a cell and place it in any other empty cell. 

If an empty cell is between two cells with water, it gets filled with 
water itself.

Find the minimum number of times he needs to perform action 1 in order to 
fill all empty cells with water. 

Note that you don't need to minimize the use of action 2. Note that 
blocked cells neither contain water nor can Filip place water in them.
*/
void run(){
    int n;string s;cin>>n>>s;
    int cur=0;
    vector<int> a;
    for(int i=0;i<n;i++){
        if(s[i]=='#'){
            if(cur)a.push_back(cur);
            cur=0;
        }else cur++;
    }
    if(cur)a.push_back(cur);
    sort(a.begin(),a.end());
    if(a.empty()){
        puts("0");
        return;
    }
    if(a.back()>=3){
        puts("2");
        return;
    }
    printf("%d\n",accumulate(a.begin(),a.end(),0));
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
