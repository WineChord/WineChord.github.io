#include<bits/stdc++.h>
using namespace std;
int main(){
    int n,k;scanf("%d%d",&n,&k);
    vector<double> a;
    a.push_back(0);
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        a.push_back((x+1)*1.0/2);
        a[i+1]+=a[i];
    }
    double res=0;
    for(int i=k;i<=n;i++)res=max(res,a[i]-a[i-k]);
    printf("%.8lf\n",res);
}