#include<bits/stdc++.h>
using namespace std;
int main(){
    int n,k;scanf("%d%d",&n,&k);
    priority_queue<int,vector<int>,greater<int>> q;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        q.push(x);
        if(q.size()>=k){
            if(q.size()>k)q.pop();
            printf("%d\n",q.top());
        }
    }
}