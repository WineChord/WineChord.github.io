#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1895/B

Educational Codeforces Round 157 (Rated for Div. 2) B. Points and Minimum Distance 

You are given a sequence of integers a of length 2n. You have to split 
these 2n integers into n pairs; each pair will represent the coordinates 
of a point on a plane. Each number from the sequence a should become the x 
or y coordinate of exactly one point. Note that some points can be equal.

After the points are formed, you have to choose a path s that starts from 
one of these points, ends at one of these points, and visits all n points 
at least once.

The length of path s is the sum of distances between all adjacent points 
on the path. In this problem, the distance between two points (x_1, y_1) 
and (x_2, y_2) is defined as |x_1-x_2| + |y_1-y_2|.

Your task is to form n points and choose a path s in such a way that the 
length of path s is minimized.
*/
#define N 220
int a[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n*2;i++){
        scanf("%d",&a[i]);
    }
    sort(a,a+2*n);
    printf("%d\n",a[2*n-1]-a[n]+a[n-1]-a[0]);
    vector<vector<int>> res(n);
    for(int i=0;i<2*n;i++){
        res[i%n].push_back(a[i]);
    }
    for(auto x:res){
        printf("%d %d\n",x[0],x[1]);
    }
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
