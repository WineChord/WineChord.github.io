// #include<bits/stdc++.h>
#include<vector>
#include<algorithm>
#include<cstdio>
using namespace std;
int dp[4][6][8];
int main(){
    dp[0][0][0]=1; // 1: win, 0: lose
    vector<vector<int>> res;
    vector<vector<int>> res0;
    for(int i=0;i<=3;i++){
        for(int j=0;j<=5;j++){
            for(int k=0;k<=7;k++){
                for(int ii=0;ii<i;ii++)if(dp[ii][j][k]==0)dp[i][j][k]=1;
                for(int ii=0;ii<j;ii++)if(dp[i][ii][k]==0)dp[i][j][k]=1;
                for(int ii=0;ii<k;ii++)if(dp[i][j][ii]==0)dp[i][j][k]=1;
                if(dp[i][j][k]==0){
                    printf("%d %d %d\n",i,j,k);
                    vector<int> a={i,j,k};
                    sort(a.begin(),a.end());
                    res.push_back(a);
                }else{
                    int v=i^j^k;
                    if(v==0){
                        vector<int> a={i,j,k};
                        sort(a.begin(),a.end());
                        res0.push_back(a);
                    }
                }
            }
        }
    }
    printf("--------------\n");
    sort(res.begin(),res.end());
    res.erase(unique(res.begin(),res.end()),res.end());
    for(auto& v:res){
        printf("%d %d %d, xor=%d\n",v[0],v[1],v[2],v[0]^v[1]^v[2]);
    }
    printf("-------------- xor=0\n");
    sort(res0.begin(),res0.end());
    res0.erase(unique(res0.begin(),res0.end()),res0.end());
    for(auto& v:res0){
        printf("%d %d %d, xor=%d\n",v[0],v[1],v[2],v[0]^v[1]^v[2]);
    }
}