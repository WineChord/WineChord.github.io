#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1883/A

Codeforces Round 905 (Div. 3) A. Morning 

You are given a four-digit pin code consisting of digits from 0 to 9 that 
needs to be entered. Initially, the cursor points to the digit 1. In one 
second, you can perform exactly one of the following two actions:

 Press the cursor to display the current digit,

 Move the cursor to any adjacent digit.

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/8764a5e2e2f2eda05775f6cbf1fbc53d7699ca
d5.png" style="max-width: 100.0%;max-height: 100.0%;" /> 

The image above shows the device you are using to enter the pin code. For 
example, for the digit 5, the adjacent digits are 4 and 6, and for the 
digit 0, there is only one adjacent digit, 9.

Determine the minimum number of seconds required to enter the given 
four-digit pin code.
*/
void run(){
    string s;cin>>s;
    int cur=0;
    string t="1234567890";
    int res=0;
    for(auto c:s){
        if(c==t[cur]){
            res++;
            continue;
        }
        int x=c-'0';
        if(x==0)x+=10;
        x--;
        res+=abs(x-cur)+1;
        cur=x;
    }
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
