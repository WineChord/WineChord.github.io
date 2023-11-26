#include<bits/stdc++.h>
using namespace std;
bool ins[30];
int main(){
    string s;cin>>s;
    int n=s.size();
    stack<char> st;
    for(int i=0;i<n;i++){
        if(s[i]==')'){
            while(st.top()!='('){
                ins[st.top()]=false;
                st.pop();
            }
            st.pop();
        }else if(s[i]=='(')st.push(s[i]);
        else{
            if(ins[s[i]]){
                puts("No");
                return 0;
            }
            st.push(s[i]);
            ins[s[i]]=true;
        }
    }
    puts("Yes");
}