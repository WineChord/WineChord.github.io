#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;string s;cin>>n>>s;
    int res=0;
    for(int st=0;st<n;st++){
        int pre=st-1;
        int at=0,cg=0;
        for(int i=st;i<n;i++){
            char c=s[i];
            if(c=='A')at++;
            if(c=='T')at--;
            if(c=='C')cg++;
            if(c=='G')cg--;
            if(at==0&&cg==0){
                res++;
                pre=i;
            }
        }
    }
    printf("%d\n",res);
}