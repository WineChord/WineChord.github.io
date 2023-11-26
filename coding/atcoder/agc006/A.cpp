#include<bits/stdc++.h>
using namespace std;
int main(){
    int n;string s,t;cin>>n>>s>>t;
    for(int i=0;i<n;i++){
        if(s.substr(i,n-i)==t.substr(0,n-i)){
            cout<<2*i+n-i<<endl;
            return 0;
        }
    }
    // abcd cdef
    cout<<2*n<<endl;
}