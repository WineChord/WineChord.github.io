#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1863/A

Pinely Round 2 (Div. 1 + Div. 2) A. Channel 

Petya is an administrator of a channel in one of the messengers. A total 
of n people are subscribed to his channel, and Petya is not considered a 
subscriber.

Petya has published a new post on the channel. At the moment of the 
publication, there were a subscribers online. We assume that every 
subscriber always reads all posts in the channel if they are online.

After this, Petya starts monitoring the number of subscribers online. He 
consecutively receives q notifications of the form "a subscriber went 
offline" or "a subscriber went online". Petya does not know which exact 
subscriber goes online or offline. It is guaranteed that such a sequence 
of notifications could have indeed been received.

Petya wonders if all of his subscribers have read the new post. Help him 
by determining one of the following: 

 it is impossible that all n subscribers have read the post; 

 it is possible that all n subscribers have read the post; 

 it is guaranteed that all n subscribers have read the post. 
*/
char s[110];
void run(){
    int n,a,q;scanf("%d%d%d",&n,&a,&q);
    scanf("%s",s);
    int mx=a;int cur=a;int plus=a;
    for(int i=0;i<q;i++){
        if(s[i]=='+')cur++,plus++;
        else cur--;
        mx=max(mx,cur);
    }
    if(mx>=n){
        puts("YES");
        return;
    }
    if(plus>=n){
        puts("MAYBE");
        return;
    }
    puts("NO");
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
