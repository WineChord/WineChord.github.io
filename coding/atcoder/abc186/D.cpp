#include<bits/stdc++.h>
#define N 200020
using namespace std;
using ll=long long;
ll a[N],s[N];
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%lld",&a[i]);
    }
    sort(a,a+n);
    // sum a[j]-a[i]
    // (n-1)*a[n-1]-s[n-2]
    // (n-2)*a[n-2]-s[n-3]
    // a[1]-s[0]
    for(int i=0;i<n;i++){
        if(i)s[i]=s[i-1];
        s[i]+=a[i];
    }
    ll res=0;
    for(int j=1;j<n;j++){
        res+=j*a[j]-s[j-1];
    }
    printf("%lld\n",res);
}