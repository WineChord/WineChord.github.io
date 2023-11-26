#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1843/C

Codeforces Round 881 (Div. 3) C. Sum in Binary Tree 

Vanya really likes math. One day when he was solving another math problem, 
he came up with an interesting tree. This tree is built as follows.

Initially, the tree has only one vertex with the number 1 — the root of 
the tree. Then, Vanya adds two children to it, assigning them consecutive 
numbers — 2 and 3, respectively. After that, he will add children to the 
vertices in increasing order of their numbers, starting from 2, assigning 
their children the minimum unused indices. As a result, Vanya will have an 
infinite tree with the root in the vertex 1, where each vertex will have 
exactly two children, and the vertex numbers will be arranged sequentially 
by layers.

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/3fe851b2505ce276dabd4a63ad7472346f98f9
a8.png" style="max-width: 100.0%;max-height: 100.0%;" />   Part of Vanya's 
tree. 

Vanya wondered what the sum of the vertex numbers on the path from the 
vertex with number 1 to the vertex with number n in such a tree is equal 
to. Since Vanya doesn't like counting, he asked you to help him find this 
sum.
*/
void run(){
    using ll=long long;
    ll n;scanf("%lld",&n);
    ll res=n;
    while(n){
        n/=2;res+=n;
    }
    printf("%lld\n",res);
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
