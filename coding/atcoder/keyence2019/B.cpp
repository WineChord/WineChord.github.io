#include<bits/stdc++.h>
using namespace std;
int main(){
    string t="keyence";
    string s;cin>>s;
    if(t==s){
        puts("YES");
        return 0;
    }
    int n=s.size();
    for(int i=0;i<n;i++)
        for(int j=i;j<n;j++){
            string x=s.substr(0,i)+s.substr(j+1,n-j-1);
            if(x==t){
                puts("YES");
                return 0;
            }
        }
    puts("NO");
}