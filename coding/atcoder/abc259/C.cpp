#include<bits/stdc++.h>
using namespace std;
int main(){
    string s,t;cin>>s>>t;
    int n=s.size(),m=t.size();
    int i=0,j=0;
    while(i<n&&j<m){
        char c1=s[i];
        char c2=t[j];
        if(c1!=c2)break;
        int ii=i,jj=j;
        while(s[ii]==c1)ii++;
        while(t[jj]==c1)jj++;
        int len1=ii-i;
        int len2=jj-j;
        if(len1!=len2&&(len1==1||len2<len1))break;
        else i=ii,j=jj;
    }
    if(i==n&&j==m)puts("Yes");
    else puts("No");
}