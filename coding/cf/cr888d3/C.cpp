#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1851/problem/C

Codeforces Round 888 (Div. 3) C. Tiles Comeback 

Vlad remembered that he had a series of n tiles and a number k. The tiles 
were numbered from left to right, and the i-th tile had colour c_i.

If you stand on the first tile and start jumping any number of tiles 
right, you can get a path of length p. The length of the path is the 
number of tiles you stood on.

Vlad wants to see if it is possible to get a path of length p such that: 

 it ends at tile with index n; 

 p is divisible by k 

 the path is divided into blocks of length exactly k each; 

 tiles in each block have the same colour, the colors in adjacent blocks 
are not necessarily different. 



For example, let n = 14, k = 3.

The colours of the tiles are contained in the array c = [\color{red}{1}, 
\color{violet}{2}, \color{red}{1}, \color{red}{1}, \color{violet}{2}, 
\color{violet}{2}, \color{green}{3}, \color{green}{3}, \color{red}{1}, 
\color{green}{3}, \color{blue}{4}, \color{blue}{4}, \color{violet}{2}, 
\color{blue}{4}]. Then we can construct a path of length 6 consisting of 2 
blocks:

\color{red}{c_1} \rightarrow \color{red}{c_3} \rightarrow \color{red}{c_4} 
\rightarrow \color{blue}{c_{11}} \rightarrow \color{blue}{c_{12}} 
\rightarrow \color{blue}{c_{14}}

All tiles from the 1-st block will have colour \color{red}{\textbf{1}}, 
from the 2-nd block will have colour \color{blue}{\textbf{4}}.

It is also possible to construct a path of length 9 in this example, in 
which all tiles from the 1-st block will have colour 
\color{red}{\textbf{1}}, from the 2-nd block will have colour 
\color{green}{\textbf{3}}, and from the 3-rd block will have colour 
\color{blue}{\textbf{4}}.
*/
#define N 200020
int c[N];
void run(){
    int n,k;scanf("%d%d",&n,&k);
    for(int i=0;i<n;i++)scanf("%d",&c[i]);
    if(k==1){
        puts("YES");
        return;
    }
    if(c[0]==c[n-1]){
        int cnt=0;
        for(int i=0;i<n;i++){
            if(c[i]==c[0])cnt++;
            if(cnt>=k){
                puts("YES");
                return;
            }
        }
        puts("NO");
        return;
    }
    int cnt=0,l=0,r=n-1;
    for(int i=0;i<n;i++){
        if(c[i]==c[0])cnt++;
        if(cnt==k){
            l=i;
            break;
        }
    }
    if(cnt<k){
        puts("NO");
        return;
    }
    cnt=0;
    for(int i=n-1;i>=0;i--){
        if(c[i]==c[n-1])cnt++;
        if(cnt==k){
            r=i;
            break;
        }
    }
    if(cnt<k){
        puts("NO");
        return;
    }
    if(l<r)puts("YES");
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
