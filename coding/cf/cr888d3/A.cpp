#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1851/problem/0

Codeforces Round 888 (Div. 3) A. Escalator Conversations 

One day, Vlad became curious about who he can have a conversation with on 
the escalator in the subway. There are a total of n passengers. The 
escalator has a total of m steps, all steps indexed from 1 to m and i-th 
step has height i \cdot k.

Vlad's height is H centimeters. Two people with heights a and b can have a 
conversation on the escalator if they are standing on different steps and 
the height difference between them is equal to the height difference 
between the steps.

For example, if two people have heights 170 and 180 centimeters, and m = 
10, k = 5, then they can stand on steps numbered 7 and 5, where the height 
difference between the steps is equal to the height difference between the 
two people: k \cdot 2 = 5 \cdot 2 = 10 = 180 - 170. There are other 
possible ways.

Given an array h of size n, where h_i represents the height of the i-th 
person. Vlad is interested in how many people he can have a conversation 
with on the escalator individually.

For example, if n = 5, m = 3, k = 3, H = 11, and h = [5, 4, 14, 18, 2], 
Vlad can have a conversation with the person with height 5 (Vlad will 
stand on step 1, and the other person will stand on step 3) and with the 
person with height 14 (for example, Vlad can stand on step 2, and the 
other person will stand on step 3). Vlad cannot have a conversation with 
the person with height 2 because even if they stand on the extreme steps 
of the escalator, the height difference between them will be 6, while 
their height difference is 9. Vlad cannot have a conversation with the 
rest of the people on the escalator, so the answer for this example is 2.
*/
void run(){
    int n,m,k,h;scanf("%d%d%d%d",&n,&m,&k,&h);
    int cnt=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        for(int a=1;a<=m;a++){
            bool flag=false;
            for(int b=1;b<=m;b++)
                if(a!=b&&a*k+h==b*k+x){cnt++;flag=true;break;}
            if(flag)break;
        }
    }
    printf("%d\n",cnt);
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
