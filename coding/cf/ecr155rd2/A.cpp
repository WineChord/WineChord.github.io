#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1879/A

Educational Codeforces Round 155 (Rated for Div. 2) A. Rigged! 

Monocarp organizes a weightlifting competition. There are n athletes 
participating in the competition, the i-th athlete has strength s_i and 
endurance e_i. The 1-st athlete is Monocarp's friend Polycarp, and 
Monocarp really wants Polycarp to win.

The competition will be conducted as follows. The jury will choose a 
positive (greater than zero) integer w, which denotes the weight of the 
barbell that will be used in the competition. The goal for each athlete is 
to lift the barbell as many times as possible. The athlete who lifts the 
barbell the most amount of times will be declared the winner (if there are 
multiple such athletes — there's no winner).

If the barbell's weight w is strictly greater than the strength of the 
i-th athlete s_i, then the i-th athlete will be unable to lift the barbell 
even one single time. Otherwise, the i-th athlete will be able to lift the 
barbell, and the number of times he does it will be equal to his endurance 
e_i.

For example, suppose there are 4 athletes with parameters s_1 = 7, e_1 = 
4; s_2 = 9, e_2 = 3; s_3 = 4, e_3 = 6; s_4 = 2, e_4 = 2. If the weight of 
the barbell is 5, then:

 the first athlete will be able to lift the barbell 4 times; 

 the second athlete will be able to lift the barbell 3 times; 

 the third athlete will be unable to lift the barbell; 

 the fourth athlete will be unable to lift the barbell. 

Monocarp wants to choose w in such a way that Polycarp (the 1-st athlete) 
wins the competition. Help him to choose the value of w, or report that it 
is impossible.
*/
#define N 110
int s[N],e[N];
void run(){
    int n;scanf("%d",&n);
    int res=0;
    for(int i=1;i<=n;i++){
        scanf("%d%d",&s[i],&e[i]);
        if(i>1&&s[i]>=s[1]&&e[i]>=e[1])res=-1;
    }
    if(res!=-1)res=s[1];
    printf("%d\n",res);
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
