#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;scanf("%d",&n);
    int r=0;
    string s;
    for(int i=0;i<n;i++){
        char c;scanf(" %c ",&c);
        if(c=='R')r++;
        s.push_back(c);
    }
    int res=0;
    for(int i=0;i<r;i++){
        res+=s[i]!='R';
    }
    printf("%d\n",res);
}