#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1872/problem/E

Codeforces Round 895 (Div. 3) E. Data Structures Fan 

You are given an array of integers a_1, a_2, \ldots, a_n, as well as a 
binary string^{\dagger} s consisting of n characters.

Augustin is a big fan of data structures. Therefore, he asked you to 
implement a data structure that can answer q queries. There are two types 
of queries:

 "1 l r" (1\le l \le r \le n) — replace each character s_i for l \le i \le 
r with its opposite. That is, replace all \texttt{0} with \texttt{1} and 
all \texttt{1} with \texttt{0}.

 "2 g" (g \in \{0, 1\}) — calculate the value of the <a 
href="https://en.wikipedia.org/wiki/Bitwise_operation#XOR">bitwise XOR

 of the numbers a_i for all indices i such that s_i = g. Note that the 
\operatorname{XOR} of an empty set of numbers is considered to be equal to 
0.



Please help Augustin to answer all the queries!

For example, if n = 4, a = [1, 2, 3, 6], s = \texttt{1001}, consider the 
following series of queries:

 "2 0" — we are interested in the indices i for which s_i = \tt{0}, since 
s = \tt{1001}, these are the indices 2 and 3, so the answer to the query 
will be a_2 \oplus a_3 = 2 \oplus 3 = 1.

 "1 1 3" — we need to replace the characters s_1, s_2, s_3 with their 
opposites, so before the query s = \tt{1001}, and after the query: s = 
\tt{0111}.

 "2 1" — we are interested in the indices i for which s_i = \tt{1}, since 
s = \tt{0111}, these are the indices 2, 3, and 4, so the answer to the 
query will be a_2 \oplus a_3 \oplus a_4 = 2 \oplus 3 \oplus 6 = 7.

 "1 2 4" — s = \tt{0111} \to s = \tt{0000}.

 "2 1" — s = \tt{0000}, there are no indices with s_i = \tt{1}, so since 
the \operatorname{XOR} of an empty set of numbers is considered to be 
equal to 0, the answer to this query is 0.



^{\dagger} A binary string is a string containing only characters 
\texttt{0} or \texttt{1}.
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
