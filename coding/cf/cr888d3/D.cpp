#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1851/problem/D

Codeforces Round 888 (Div. 3) D. Prefix Permutation Sums 

Your friends have an array of n elements, calculated its array of prefix 
sums and passed it to you, accidentally losing one element during the 
transfer. Your task is to find out if the given array can matches 
permutation.

A permutation of n elements is an array of n numbers from 1 to n such that 
each number occurs exactly one times in it.

The array of prefix sums of the array a — is such an array b that b_i = 
\sum_{j=1}^i a_j, 1 \le i \le n.

For example, the original permutation was [1, 5, 2, 4, 3]. Its array of 
prefix sums — [1, 6, 8, 12, 15]. Having lost one element, you can get, for 
example, arrays [6, 8, 12, 15] or [1, 6, 8, 15].

It can also be shown that the array [1, 2, 100] does not correspond to any 
permutation.
*/
#define N 200020
ll a[N],b[N],c[N];
void run(){
    ll n;scanf("%lld",&n);
    ll sum=n;
    vector<ll> big;
    memset(a,0,sizeof(a));
    memset(b,0,sizeof(b));
    memset(c,0,sizeof(c));
    for(ll i=0;i<n-1;i++){
        sum+=1+i;
        scanf("%lld",&a[i]);
        if(i==0)b[i]=a[i];
        else b[i]=a[i]-a[i-1];
        if(b[i]<=n)c[b[i]]++;
        else big.push_back(b[i]);
    }
    if(big.size()>1){
        puts("NO");
        return;
    }
    if(sum!=a[n-2]){
        vector<ll> no;
        for(ll i=1;i<=n;i++){
            if(c[i]==0)no.push_back(i);
            else if(c[i]!=1){
                puts("NO");
                return;
            }
            if(no.size()>1){
                puts("NO");
                return;
            }
        }
        if(no.size()==1)puts("YES");
        else puts("NO");
        return;
    }
    if(big.size()==1){
        ll t=big[0];ll v=0,cnt=0;
        for(ll i=1;i<=n;i++){
            if(c[i]==0)v+=i,cnt++;
            else if(c[i]!=1){
                puts("NO");
                return;
            }
            if(cnt>2||v>t){
                puts("NO");
                return;
            }
        }
        if(cnt==2&&v==t)puts("YES");
        else puts("NO");
        return;
    }
    vector<ll> no,more;
    for(ll i=1;i<=n;i++){
        if(c[i]==0)no.push_back(i);
        else if(c[i]==2)more.push_back(i);
        else if(c[i]!=1){
            puts("NO");
            return;
        }
        if(no.size()>2||more.size()>1){
            puts("NO");
            return;
        }
        if(no.size()==2&&more.size()==1&&no[0]+no[1]!=more[0]){
            puts("NO");
            return;
        }
    }
    if(no.size()==2&&more.size()==1&&no[0]+no[1]==more[0])puts("YES");
    else puts("NO");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
