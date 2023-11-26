#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1843/A

Codeforces Round 881 (Div. 3) A. Sasha and Array Coloring 

Sasha found an array a consisting of n integers and asked you to paint 
elements.

You have to paint each element of the array. You can use as many colors as 
you want, but each element should be painted into exactly one color, and 
for each color, there should be at least one element of that color.

The cost of one color is the value of \max(S) - \min(S), where S is the 
sequence of elements of that color. The cost of the whole coloring is the 
sum of costs over all colors.

For example, suppose you have an array a = [\color{red}{1}, 
\color{red}{5}, \color{blue}{6}, \color{blue}{3}, \color{red}{4}], and you 
painted its elements into two colors as follows: elements on positions 1, 
2 and 5 have color 1; elements on positions 3 and 4 have color 2. Then:

 the cost of the color 1 is \max([1, 5, 4]) - \min([1, 5, 4]) = 5 - 1 = 4; 

 the cost of the color 2 is \max([6, 3]) - \min([6, 3]) = 6 - 3 = 3; 

 the total cost of the coloring is 7. 

For the given array a, you have to calculate the maximum possible cost of 
the coloring.
*/
#define N 55
int a[N];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    sort(a,a+n);
    int res=0;
    for(int i=0;i<n/2;i++){
        res+=a[n-1-i]-a[i];
    }
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
