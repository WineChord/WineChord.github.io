#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    int n;scanf("%d",&n);
    vector<int> a;
    for(int i=0;i<3*n;i++){
        int x;scanf("%d",&x);
        a.push_back(x);
    }
    sort(a.rbegin(),a.rend());
    ll res=0;
    for(int i=1;i<2*n;i+=2)res+=a[i];
    printf("%lld\n",res);
}