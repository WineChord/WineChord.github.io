#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    string s;cin>>s;
    ll k;cin>>k;
    for(auto c:s){
        if(c=='1')k--;
        else{
            printf("%c\n",c);
            return 0;
        }
        if(k==0){
            printf("1\n");
            break;
        }
    }
}