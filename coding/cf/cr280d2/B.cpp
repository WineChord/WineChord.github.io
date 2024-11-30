#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/492/B

Codeforces Round 280 (Div. 2) B. Vanya and Lanterns 

Vanya walks late at night along a straight street of length l, lit by n 
lanterns. Consider the coordinate system with the beginning of the street 
corresponding to the point 0, and its end corresponding to the point l. 
Then the i-th lantern is at the point a_i. The lantern lights all points 
of the street that are at the distance of at most d from it, where d is 
some positive number, common for all lanterns. 

Vanya wonders: what is the minimum light radius d should the lanterns have 
to light the whole street?
*/
void run(){
    int n,l;
    cin>>n>>l;
    vector<int> a(n);
    for(int i=0;i<n;i++)cin>>a[i];
    sort(a.begin(),a.end());
    double mx=max(a[0],l-a[n-1]);
    for(int i=1;i<n;i++){
        mx=max(mx,double(a[i]-a[i-1])/2);
    }
    cout<<fixed<<setprecision(10)<<mx<<endl;
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
