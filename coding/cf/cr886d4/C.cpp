#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1850/C

Codeforces Round 886 (Div. 4) C. Word on the Paper 

On an 8 \times 8 grid of dots, a word consisting of lowercase Latin 
letters is written vertically in one column, from top to bottom. What is 
it?
*/
char s[9][9];
void run(){
    for(int i=0;i<8;i++)
        scanf("%s",s[i]);
    int i=0,j=0;
    for(i=0;i<8;i++){
        for(j=0;j<8;j++)
            if(s[i][j]!='.')break;
        if(j<8&&s[i][j]!='.')break;
    }
    string res;
    for(int k=i;k<8;k++)
        if(s[k][j]!='.')res.push_back(s[k][j]);
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
