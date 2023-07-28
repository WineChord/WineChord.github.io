#include<bits/stdc++.h>
using namespace std;
using ll=long long;
int main(){
    int n=100;
    for(int i=1;i<n;i++){
        for(int j=0;j<6;j++){
            if((i>>j)&1)printf("+");
            else printf(" ");
        }
        puts("");
    }
    return 0;
}