#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;scanf("%d",&n);
    int cnt=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(x%2)cnt++;
    }
    if(cnt%2==0)puts("YES");
    else puts("NO");
}