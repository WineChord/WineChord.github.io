#include<bits/stdc++.h>
#define N 200020
using namespace std;
using pii=pair<int,int>;
int x[N],y[N];
char s[N];
int main(){
    int n;scanf("%d",&n);
    unordered_map<int,int> mpl;
    unordered_map<int,int> mpr;
    for(int i=1;i<=n;i++){
        scanf("%d%d",&x[i],&y[i]);
    }
    scanf("%s",s+1);
    for(int i=1;i<=n;i++){
        if(s[i]=='L'){
            if(mpl.find(y[i])==mpl.end()){
                mpl[y[i]]=x[i];
            }else mpl[y[i]]=max(mpl[y[i]],x[i]);
        }else{
            if(mpr.find(y[i])==mpr.end()){
                mpr[y[i]]=x[i];
            }else mpr[y[i]]=min(mpr[y[i]],x[i]);
        }
    }
    for(auto &[k,v]:mpl){
        if(mpr.find(k)==mpr.end())continue;
        if(v>=mpr[k]){
            puts("Yes");
            return 0;
        }
    }
    puts("No");
}