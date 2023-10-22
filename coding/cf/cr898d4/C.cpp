#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1873/problem/C

Codeforces Round 898 (Div. 4) C. Target Practice 

A 10 \times 10 target is made out of five "rings" as shown. Each ring has 
a different point value: the outermost ring — 1 point, the next ring — 2 
points, ..., the center ring — 5 points.

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/5c51151adbfdd4a0384d7616d0dc23fe1a2125
50.png" style="max-width: 100.0%;max-height: 100.0%;" /> 

Vlad fired several arrows at the target. Help him determine how many 
points he got.
*/
void run(){
    int res=0;
    for(int i=0;i<10;i++){
        string s;cin>>s;
        for(int j=0;j<10;j++){
            char c=s[j];
            // scanf(" %c ",&c);
            if(c=='X'){
                int x=i,y=j;
                if(x>4)x=9-x;
                if(y>4)y=9-y;
                res+=min(x,y)+1;
            }
        }
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
