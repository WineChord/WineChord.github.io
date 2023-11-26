#include<bits/stdc++.h>
using namespace std;
int main(){
    string s;cin>>s;
    string res="";
    for(auto c:s){
        if(c!='B')res.push_back(c);
        else if(res.size())res.pop_back();
    }
    cout<<res<<endl;
}