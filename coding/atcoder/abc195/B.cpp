#include<bits/stdc++.h>
using namespace std;
int main(){
    int a,b,w;cin>>a>>b>>w;
    int cnt=(w*1000+b-1)/b;
    if(w*1000/cnt<a){
        puts("UNSATISFIABLE");
        return 0;
    }
    int cnt2=(w*1000)/a;
    if((w*1000+cnt2-1)/cnt2>b){
        puts("UNSATISFIABLE");
        return 0;
    }
    cout<<cnt<<" "<<cnt2<<endl;
}