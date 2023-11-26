#include<bits/stdc++.h>
#define N 10
using namespace std;
string s[N];
int in[N],fa[N];
int find(int x){
    if(fa[x]==-1)return x;
    return fa[x]=find(fa[x]);
}
int main(){
    memset(fa,-1,sizeof(fa));
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)cin>>s[i];
    auto dis=[&](string& a,string& b){
        int cnt=0;
        for(int i=0;i<m;i++)if(a[i]!=b[i])cnt++;
        return cnt;
    };
    sort(s,s+n);
    do{
        bool flag=true;
        for(int i=1;i<n;i++)if(dis(s[i],s[i-1])!=1){
            flag=false;break;
        }
        if(flag==true){
            puts("Yes");return 0;
        }
    }while(next_permutation(s,s+n));
    puts("No");
}