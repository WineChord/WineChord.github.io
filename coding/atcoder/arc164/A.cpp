#include<bits/stdc++.h>
using namespace std;
using ll=long long;
void run(){
    ll n,k;cin>>n>>k;
    ll cnt=0;
    while(n){
        cnt+=n%3;
        n/=3;
    }
    if(cnt>k){
        puts("No");
        return;
    }
    if(cnt%2==k%2){
        puts("Yes");
        return;
    }
    puts("No");
}
int main(){
    int T;cin>>T;
    while(T--)run();
}