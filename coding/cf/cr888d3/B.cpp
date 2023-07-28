#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/contest/1851/problem/B

Codeforces Round 888 (Div. 3) B. Parity Sort 

You have an array of integers a of length n. You can apply the following 
operation to the given array: 

 Swap two elements a_i and a_j such that i \neq j, a_i and a_j are either 
both even or both odd. 



 Determine whether it is possible to sort the array in non-decreasing 
order by performing the operation any number of times (possibly zero).

For example, let a = [7, 10, 1, 3, 2]. Then we can perform 3 operations to 
sort the array: 

 Swap a_3 = 1 and a_1 = 7, since 1 and 7 are odd. We get a = [1, 10, 7, 3, 
2]; 

 Swap a_2 = 10 and a_5 = 2, since 10 and 2 are even. We get a = [1, 2, 7, 
3, 10]; 

 Swap a_4 = 3 and a_3 = 7, since 3 and 7 are odd. We get a = [1, 2, 3, 7, 
10]. 
*/
#define N 200020
int a[N];
void run(){
    int n;scanf("%d",&n);
    vector<int> odd,even;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(a[i]%2)odd.push_back(a[i]),a[i]=1;
        else even.push_back(a[i]),a[i]=0;
    }
    sort(odd.begin(),odd.end());
    sort(even.begin(),even.end());
    int i=0,j=0,idx=0;
    while(i<odd.size()||j<even.size()){
        if(a[idx]==1)a[idx++]=odd[i++];
        else a[idx++]=even[j++];
        if(idx>1&&a[idx-1]<a[idx-2]){
            puts("NO");
            return;
        }
    }
    puts("YES");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
