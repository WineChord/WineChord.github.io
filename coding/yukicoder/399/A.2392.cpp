#include<bits/stdc++.h>
using namespace std;
int main(){
    int num;scanf("%d",&num);
    for(int i=1;i*i<num;i++){
        int x=i*i;
        for(int j=1;j<=i;j++){
            int y=j*j;
            if(x+y>num)break;
            if(i*i+j*j==num){
                puts("Yes");
                return 0;
            }
        }
    }
    puts("No");
}