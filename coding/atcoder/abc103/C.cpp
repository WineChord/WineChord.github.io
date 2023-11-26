#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;scanf("%d",&n);
    int res=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        res+=x-1;
    }
    printf("%d\n",res);
}