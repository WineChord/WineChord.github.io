#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1901/A

Educational Codeforces Round 158 (Rated for Div. 2) A. Line Trip 

There is a road, which can be represented as a number line. You are 
located in the point 0 of the number line, and you want to travel from the 
point 0 to the point x, and back to the point 0.

You travel by car, which spends 1 liter of gasoline per 1 unit of distance 
travelled. When you start at the point 0, your car is fully fueled (its 
gas tank contains the maximum possible amount of fuel).

There are n gas stations, located in points a_1, a_2, \dots, a_n. When you 
arrive at a gas station, you fully refuel your car. Note that you can 
refuel only at gas stations, and there are no gas stations in points 0 and 
x.

You have to calculate the minimum possible volume of the gas tank in your 
car (in liters) that will allow you to travel from the point 0 to the 
point x and back to the point 0.
*/
#define N 55
int a[N];
void run(){
    int n,x;cin>>n>>x;
    for(int i=0;i<n;i++){
        cin>>a[i];
    }
    int res=a[0];
    for(int i=1;i<n;i++){
        res=max(res,a[i]-a[i-1]);
    }
    res=max(res,2*(x-a[n-1]));
    cout<<res<<endl;
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
