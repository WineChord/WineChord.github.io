// 2
#include<bits/stdc++.h>
#define MAXN 1000010
using namespace std;
int notp[MAXN],ps[MAXN],cnt;
int getp(int n){
    for(int i=2;i<=n;i++){
        if(notp[i])continue;
        ps[cnt++]=i;
        for(int j=i+i;j<=n;j+=i)notp[j]=1;
    }
}
int main(){
    int n;cin>>n;
    getp(n);
    cout<<cnt<<endl;
}