#include<bits/stdc++.h>
using namespace std;
int main(){
    string s;cin>>s;
    unordered_map<char,int> mp;
    for(auto c:s)mp[c]++;
    if(
        mp['N']&&!mp['S']||
        mp['S']&&!mp['N']||
        mp['E']&&!mp['W']||
        mp['W']&&!mp['E']
    ){
        puts("No");
    }else puts("Yes");
}