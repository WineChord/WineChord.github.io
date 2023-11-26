#include<bits/stdc++.h>
#define N 110
using namespace std;
int a[N];
int main(){
    int n;scanf("%d",&n);
    int zero=0;
    int tot=0;
    vector<int> can;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        zero+=a[i]%10==0;
        tot+=a[i];
        if(a[i]%10)can.push_back(a[i]);
    }
    sort(can.begin(),can.end());
    if(zero==n){
        puts("0");
        return 0;
    }
    if(tot%10!=0){
        printf("%d\n",tot);
        return 0;
    }
    printf("%d\n",tot-can[0]);
}