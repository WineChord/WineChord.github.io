#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1900/problem/E

Codeforces Round 911 (Div. 2) E. Transitive Graph 

You are given a directed graph G with n vertices and m edges between them.

Initially, graph H is the same as graph G. Then you decided to perform the 
following actions: 

 If there exists a triple of vertices a, b, c of H, such that there is an 
edge from a to b and an edge from b to c, but there is no edge from a to 
c, add an edge from a to c. 

 Repeat the previous step as long as there are such triples. 



Note that the number of edges in H can be up to n^2 after performing the 
actions.

You also wrote some values on vertices of graph H. More precisely, vertex 
i has the value of a_i written on it.

Consider a simple path consisting of k distinct vertices with indexes v_1, 
v_2, \ldots, v_k. The length of such a path is k. The value of that path 
is defined as \sum_{i = 1}^k a_{v_i}.

A simple path is considered the longest if there is no other simple path 
in the graph with greater length.

Among all the longest simple paths in H, find the one with the smallest 
value.
*/
void run(){
    // Welcome, your majesty.
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
