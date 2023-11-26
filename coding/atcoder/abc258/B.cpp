#include<bits/stdc++.h>
#define N 11
using namespace std;
int a[N][N];
int dx[8]={-1, 0, 1,-1,1,-1,0,1};
int dy[8]={-1,-1,-1, 0,0, 1,1,1};
int main(){
    int n;cin>>n;
    for(int i=0;i<n;i++){
        string s;cin>>s;
        for(int j=0;j<n;j++){
            a[i][j]=s[j]-'0';
        }
    }
    auto get=[&](int& x,int& y,int d){
        int nx=n+x+dx[d];
        int ny=n+y+dy[d];
        nx%=n;ny%=n;
        x=nx;y=ny;
        return a[nx][ny];
    };
    int x=0,y=0,mx=a[0][0];
    using pii=pair<int,int>;
    vector<pii> can;
    can.push_back({x,y});
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            if(a[i][j]>mx){
                mx=a[i][j];
                x=i;y=j;
                can=vector<pii>();
                can.push_back({x,y});
            }else if(a[i][j]==mx){
                x=i;y=j;
                can.push_back({x,y});
            }
        }
    int sz=can.size();
    vector<vector<int>> ca;
    for(auto [x,y]:can){
        vector<vector<int>> res(8);
        for(int d=0;d<8;d++){
            int xx=x,yy=y;
            // printf("# %d: ",d);
            for(int i=0;i<n;i++){
                res[d].push_back(a[xx][yy]);
                // printf("%d ",res[d].back());
                get(xx,yy,d);
            }
            // printf("\n");
        }
        sort(res.begin(),res.end());
        ca.push_back(res[7]);
    }
    sort(ca.begin(),ca.end());
    for(int i=0;i<n;i++){
        cout<<ca[sz-1][i];
    }
    cout<<endl;
}