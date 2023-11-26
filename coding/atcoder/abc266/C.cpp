#include<bits/stdc++.h>
using namespace std;
int main(){
    using pii=pair<int,int>;
    vector<pii> a;
    for(int i=0;i<4;i++){
        int x,y;cin>>x>>y;
        a.push_back({x,y});
    }
    auto line=[&](pii a,pii b){
        auto [x1,y1]=a;
        auto [x2,y2]=b;
        return pii{x2-x1,y2-y1};
    };
    auto dot=[&](pii l1,pii l2){
        auto [x1,y1]=l1;
        auto [x2,y2]=l2;
        return x1*y2-y1*x2;
    };
    for(int i=0;i<4;i++){
        auto p1=a[i];
        auto p2=a[(i+1+4)%4];
        auto p3=a[(i+2+4)%4];
        auto l1=line(p1,p2);
        auto l2=line(p2,p3);
        if(dot(l1,l2)<=0){
            puts("No");
            return 0;
        }
    }
    puts("Yes");
}