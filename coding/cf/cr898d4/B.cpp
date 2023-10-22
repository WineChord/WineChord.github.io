#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1873/problem/B

Codeforces Round 898 (Div. 4) B. Good Kid 

Slavic is preparing a present for a friend's birthday. He has an array a 
of n digits and the present will be the product of all these digits. 
Because Slavic is a good kid who wants to make the biggest product 
possible, he wants to add 1 to exactly one of his digits. 

What is the maximum product Slavic can make?
*/
void run(){
    int n;scanf("%d",&n);
    vector<int> a;
    for(int i=0;i<n;i++){
        int x;
        scanf("%d",&x);
        a.push_back(x);
    }
    sort(a.begin(),a.end());
    a[0]++;
    int res=1;
    for(auto x:a)res*=x;
    printf("%d\n",res);
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
