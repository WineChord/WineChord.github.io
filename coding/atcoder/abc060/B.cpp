#include<bits/stdc++.h>
using namespace std;
int main(){
    unordered_set<int> s;
    int a,b,c;scanf("%d%d%d",&a,&b,&c);
    int x=a%b;
    while(s.count(x)==0){
        s.insert(x);
        x=(x+a)%b;
        if(s.count(c)){
            puts("YES");
            return 0;
        }
    }
    puts("NO");
}