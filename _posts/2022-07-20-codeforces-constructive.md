---
classes: splash
title: "Codeforces Constructive Problems"
excerpt: "Practices of some codeforces constructive problems."
categories: 
  - coding
tags: 
  - constructive
toc: true
toc_sticky: true
---

# 1. Codeforces Round #720 (Div. 2) A. Nastia and Nearly Good Numbers

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1521/A

Codeforces Round #720 (Div. 2) A. Nastia and Nearly Good Numbers

Nastia has 2 positive integers A and B. She defines that:

The integer is good if it is divisible by A⋅B;
Otherwise, the integer is nearly good, if it is divisible by A.

Find 3 different positive integers x, y, and z such that exactly one of 
them is good and the other 2 are nearly good, and x+y=z.
*/
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        ll a,b;scanf("%lld%lld",&a,&b);
        if(b==1)printf("NO\n");
        else printf("YES\n%lld %lld %lld\n",(3*b-1)*a,a,3*a*b);
    }
}
```

# 2. Educational Codeforces Round 103 A. K-divisible Sum

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1476/A

Educational Codeforces Round 103 A. K-divisible Sum

You are given two integers n and k.

You should create an array of n positive integers a1,a2,…,an 
such that the sum (a1+a2+⋯+an) is divisible by k and maximum 
element in a is minimum possible.

What is the minimum possible maximum element in a?
*/
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        ll n,k;scanf("%lld%lld",&n,&k);
        if(k<n)k=k*((n-1)/k+1);
        printf("%lld\n",(k-1)/n+1);
    }
}
```

# 3. Codeforces Round #719 (Div. 3) C. Not Adjacent Matrix

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1520/C

Codeforces Round #719 (Div. 3) C. Not Adjacent Matrix

We will consider the numbers a and b as adjacent if they differ by 
exactly one, that is, |a−b|=1.

We will consider cells of a square matrix n×n as adjacent if they have 
a common side, that is, for cell (r,c) cells (r,c−1), (r,c+1), (r−1,c) 
and (r+1,c) are adjacent to it.

For a given number n, construct a square matrix n×n such that:

Each integer from 1 to n2 occurs in this matrix exactly once;
If (r1,c1) and (r2,c2) are adjacent cells, then the numbers written in 
them must not be adjacent.
*/
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        ll n,k;scanf("%lld%lld",&n,&k);
        if(k<n)k=k*((n-1)/k+1);
        printf("%lld\n",(k-1)/n+1);
    }
}
```

# 4. Codeforces Global Round 7 A. Bad Ugly Numbers 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1326/A

Codeforces Global Round 7 A. Bad Ugly Numbers 

You are given a integer n (n > 0). Find any integer s which satisfies 
these conditions, or report that there are no such numbers:

In the decimal representation of s: 

 s > 0, 

 s consists of n digits, 

 no digit in s equals 0, 

 s is not divisible by any of it's digits. 
*/
void run(){
    int n;scanf("%d",&n);
    if(n==1){
        printf("-1\n");
        return;
    }
    printf("2");
    for(int i=0;i<n-1;i++){
        printf("3");
    }
    printf("\n");
    return;
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
```

# 5. Codeforces Round #360 (Div. 2) B. Lovely Palindromes 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/688/B

Codeforces Round #360 (Div. 2) B. Lovely Palindromes 

Pari has a friend who loves palindrome numbers. A palindrome number is a 
number that reads the same forward or backward. For example 12321, 100001 
and 1 are palindrome numbers, while 112 and 1021 are not.

Pari is trying to love them too, but only very special and gifted people 
can understand the beauty behind palindrome numbers. Pari loves integers 
with even length (i.e. the numbers with even number of digits), so she 
tries to see a lot of big palindrome numbers with even length (like a 
2-digit 11 or 6-digit 122221), so maybe she could see something in them.

Now Pari asks you to write a program that gets a huge integer n from the 
input and tells what is the n-th even-length positive palindrome number?
*/
void run(){
    string s;
    cin>>s;
    cout<<s;
    reverse(s.begin(),s.end());
    cout<<s<<"\n";
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 6. Codeforces Round #396 (Div. 2) B. Mahmoud and a Triangle 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/766/B

Codeforces Round #396 (Div. 2) B. Mahmoud and a Triangle 

Mahmoud has n line segments, the i-th of them has length a_i. Ehab 
challenged him to use exactly 3 line segments to form a non-degenerate 
triangle. Mahmoud doesn't accept challenges unless he is sure he can win, 
so he asked you to tell him if he should accept the challenge. Given the 
lengths of the line segments, check if he can choose exactly 3 of them to 
form a non-degenerate triangle.

Mahmoud should use exactly 3 line segments, he can't concatenate two line 
segments or change any length. A non-degenerate triangle is a triangle 
with positive area.
*/
#define MAXN 300030
ll a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%lld",&a[i]);
    sort(a,a+n);
    for(int i=2;i<n;i++){
        if(a[i-2]+a[i-1]>a[i]){
            printf("YES\n");
            return;
        }
    }
    printf("NO\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 7. Codeforces Beta Round #89 (Div. 2) B. Present from Lena 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/118/B

Codeforces Beta Round #89 (Div. 2) B. Present from Lena 

Vasya's birthday is approaching and Lena decided to sew a patterned 
handkerchief to him as a present. Lena chose digits from 0 to n as the 
pattern. The digits will form a rhombus. The largest digit n should be 
located in the centre. The digits should decrease as they approach the 
edges. For example, for n = 5 the handkerchief pattern should look like 
that: 


          0
        0 1 0
      0 1 2 1 0
    0 1 2 3 2 1 0
  0 1 2 3 4 3 2 1 0
0 1 2 3 4 5 4 3 2 1 0
  0 1 2 3 4 3 2 1 0
    0 1 2 3 2 1 0
      0 1 2 1 0
        0 1 0
          0


Your task is to determine the way the handkerchief will look like by the 
given n.
*/
void run(){
    int n;scanf("%d",&n);
    auto print=[&](int i){
        for(int j=0;j<2*(n+i+1)-1;j++){
            if(j%2||j-2*(n-i)<0)printf(" ");
            else {
                int x=(j-2*(n-i))/2;
                if(j>2*n){
                    x=i-(x-i);
                    printf("%d",x);
                }else printf("%d",x);
            }
        }
        printf("\n");
    };
    for(int i=0;i<=n;i++){
        print(i);
    }
    for(int i=n-1;i>=0;i--){
        print(i);
    }
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 8. Educational Codeforces Round 101 (Rated for Div. 2) A. Regular Bracket Sequence 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1469/A

Educational Codeforces Round 101 (Rated for Div. 2) A. Regular Bracket Sequence 

A bracket sequence is called regular if it is possible to obtain correct 
arithmetic expression by inserting characters + and 1 into this sequence. 
For example, sequences (())(), () and (()(())) are regular, while )(, (() 
and (()))( are not. Let's call a regular bracket sequence "RBS".

You are given a sequence s of n characters (, ), and/or ?. There is 
exactly one character ( and exactly one character ) in this sequence.

You have to replace every character ? with either ) or ( (different 
characters ? can be replaced with different brackets). You cannot reorder 
the characters, remove them, insert other characters, and each ? must be 
replaced.

Determine if it is possible to obtain an RBS after these replacements.
*/
void run(){
    string s;cin>>s;
    int n=s.length();
#define no do{printf("NO\n");return;}while(0)
#define yes do{printf("YES\n");return;}while(0)
    if(n%2)no;
    int l=0,r=0;
    for(int i=0;i<n;i++){
        if(s[i]=='(')l++;
        if(s[i]==')')r++;
    }
    if(l>n/2||r>n/2)no;
    l=n/2-l;r=n/2-r;
    for(int i=0;i<n;i++){
        if(s[i]!='?')continue;
        if(l)s[i]='(',l--;
        else s[i]=')';
    }
    l=0;r=0;
    for(int i=0;i<n;i++){
        if(s[i]=='(')l++;
        else r++;
        if(r>l)no;
    }
    yes;
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
```

# 9. Codeforces Round #396 (Div. 2) A. Mahmoud and Longest Uncommon Subsequence 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/766/A

Codeforces Round #396 (Div. 2) A. Mahmoud and Longest Uncommon Subsequence 

While Mahmoud and Ehab were practicing for IOI, they found a problem which 
name was Longest common subsequence. They solved it, and then Ehab 
challenged Mahmoud with another problem.

Given two strings a and b, find the length of their longest uncommon 
subsequence, which is the longest string that is a subsequence of one of 
them and not a subsequence of the other.

A subsequence of some string is a sequence of characters that appears in 
the same order in the string, The appearances don't have to be 
consecutive, for example, strings "ac", "bc", "abc" and "a" are 
subsequences of string "abc" while strings "abbc" and "acb" are not. The 
empty string is a subsequence of any string. Any string is a subsequence 
of itself.
*/
void run(){
    string a,b;
    cin>>a>>b;
    if(a==b){
        printf("-1\n");
        return;
    }
    int n=a.length(),m=b.length();
    printf("%d\n",max(n,m));
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 10. Codeforces Round #702 (Div. 3) B. Balanced Remainders 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1490/B

Codeforces Round #702 (Div. 3) B. Balanced Remainders 

You are given a number n (divisible by 3) and an array a[1 \dots n]. In 
one move, you can increase any of the array elements by one. Formally, you 
choose the index i (1 \le i \le n) and replace a_i with a_i + 1. You can 
choose the same index i multiple times for different moves.

Let's denote by c_0, c_1 and c_2 the number of numbers from the array a 
that have remainders 0, 1 and 2 when divided by the number 3, 
respectively. Let's say that the array a has balanced remainders if c_0, 
c_1 and c_2 are equal.

For example, if n = 6 and a = [0, 2, 5, 5, 4, 8], then the following 
sequence of moves is possible: 

 initially c_0 = 1, c_1 = 1 and c_2 = 4, these values are not equal to 
each other. Let's increase a_3, now the array a = [0, 2, 6, 5, 4, 8]; 

 c_0 = 2, c_1 = 1 and c_2 = 3, these values are not equal. Let's increase 
a_6, now the array a = [0, 2, 6, 5, 4, 9]; 

 c_0 = 3, c_1 = 1 and c_2 = 2, these values are not equal. Let's increase 
a_1, now the array a = [1, 2, 6, 5, 4, 9]; 

 c_0 = 2, c_1 = 2 and c_2 = 2, these values are equal to each other, which 
means that the array a has balanced remainders. 



Find the minimum number of moves needed to make the array a have balanced 
remainders.
*/
void run(){
    int n;scanf("%d",&n);
    vector<int> c(3,0);
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        c[x%3]++;
    }
    int res=0,t=n/3;
    for(int i=0;i<6;i++){
        int d=c[i%3]-t;
        if(d>0){
            res+=d;
            c[i%3]-=d;
            c[(i+1)%3]+=d;
        }
    }
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
```

# 11. Codeforces Round #632 (Div. 2) A. Little Artem 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1333/A

Codeforces Round #632 (Div. 2) A. Little Artem 

Young boy Artem tries to paint a picture, and he asks his mother Medina to 
help him. Medina is very busy, that's why she asked for your help.

Artem wants to paint an n \times m board. Each cell of the board should be 
colored in white or black. 

Lets B be the number of black cells that have at least one white neighbor 
adjacent by the side. Let W be the number of white cells that have at 
least one black neighbor adjacent by the side. A coloring is called good 
if B = W + 1. 

The first coloring shown below has B=5 and W=4 (all cells have at least 
one neighbor with the opposite color). However, the second coloring is not 
good as it has B=4, W=4 (only the bottom right cell doesn't have a 
neighbor with the opposite color).

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/5611f0803c61268019d39e2781107538b4c56aeb.png" style="max-width: 100.0%;max-height: 100.0%;" /> 

Please, help Medina to find any good coloring. It's guaranteed that under 
given constraints the solution always exists. If there are several 
solutions, output any of them.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    /*
    BWBWB
    BWBWW
    BWBWB
    BWBWW
    BWBWB
    */
    if(n%2&&m%2){
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(j==m-1){
                    if(i%2)printf("W");
                    else printf("B");
                }else{
                    if(j%2)printf("W");
                    else printf("B");
                }
            }
            printf("\n");
        }
        return;
    }
    /*
    BBBBB
    BWWWW
    BBBBB
    WWWWW
    */
    if(n%2==0){
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(i==1&&j==0)printf("B");
                else{
                    if(i%2)printf("W");
                    else printf("B");
                }
            }
            printf("\n");
        }
        return;
    }
    /*
    BBBW
    BWBW
    BWBW
    BWBW
    BWBW
    */
    if(m%2==0){
        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(i==0&&j==1)printf("B");
                else{
                    if(j%2)printf("W");
                    else printf("B");
                }
            }
            printf("\n");
        }
        return;
    }
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
```

# 12. Codeforces Round #696 (Div. 2) B. Different Divisors 

## Method 1: Prime Checking.

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1474/B

Codeforces Round #696 (Div. 2) B. Different Divisors 

Positive integer x is called divisor of positive integer y, if y is 
divisible by x without remainder. For example, 1 is a divisor of 7 and 3 
is not divisor of 8.

We gave you an integer d and asked you to find the smallest positive 
integer a, such that 

 a has at least 4 divisors; 

 difference between any two divisors of a is at least d.
*/
void run(){
    int d;cin>>d;
    auto isp=[](ll n){
        for(int i=2;i*i<=n;i++)if(n%i==0)return false;
        return true;
    };
    ll a=d+1;
    while(!isp(a))a++;
    ll b=a+d;
    while(!isp(b))b++;
    cout<<a*b<<endl;
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
```

## Method 2: Save Next Prime.

```cpp
#include<bits/stdc++.h>
#define MAXD 10000
#define MAXN (MAXD*2+1000)
using namespace std;
using ll=long long;
bool isp[MAXN];
int nxtp[MAXN];
void init(){
    memset(isp,true,sizeof(isp));
    // Eratosthenes Sieve.
    isp[0]=isp[1]=false;
    for(int i=2;i<MAXN;i++)if(isp[i])for(int j=i*i;j<MAXN;j+=i)isp[j]=false;
    for(int i=MAXN-2;i>=2;i--)isp[i]?nxtp[i]=i:nxtp[i]=nxtp[i+1];
}
void run(){
    int d;scanf("%d",&d);
    ll a=nxtp[1+d];
    ll b=nxtp[a+d];
    printf("%lld\n",a*b);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    init();
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
```

## Method 3: Binary Search on Saved Primes.

```cpp
#include<bits/stdc++.h>
#define MAXD 10000
#define MAXN (MAXD*2+1000)
using namespace std;
using ll=long long;
bool isp[MAXN];
vector<int> ps;
void init(){
    memset(isp,true,sizeof(isp));
    // Eratosthenes Sieve.
    isp[0]=isp[1]=false;
    for(int i=2;i<MAXN;i++)if(isp[i]){
        ps.push_back(i);
        for(int j=i*i;j<MAXN;j+=i)isp[j]=false;
    }
}
void run(){
    int d;scanf("%d",&d);
    ll a=*lower_bound(ps.begin(),ps.end(),d+1);
    ll b=*lower_bound(ps.begin(),ps.end(),a+d);
    printf("%lld\n",a*b);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    init();
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}
```

# 13. Educational Codeforces Round 83 (Rated for Div. 2) B. Bogosort 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1312/B

Educational Codeforces Round 83 (Rated for Div. 2) B. Bogosort 

You are given an array a_1, a_2, \dots , a_n. Array is good if for each 
pair of indexes i &lt; j the condition j - a_j \ne i - a_i holds. Can you 
shuffle this array so that it becomes good? To shuffle an array means to 
reorder its elements arbitrarily (leaving the initial order is also an 
option).

For example, if a = [1, 1, 3, 5], then shuffled arrays [1, 3, 5, 1], [3, 
5, 1, 1] and [5, 3, 1, 1] are good, but shuffled arrays [3, 1, 5, 1], [1, 
1, 3, 5] and [1, 1, 5, 3] aren't.

It's guaranteed that it's always possible to shuffle an array to meet this 
condition.
*/
#define MAXN 110
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
    }
    sort(a,a+n,[](int x,int y){return x>y;});
    for(int i=0;i<n;i++)printf("%d%c",a[i]," \n"[i==n-1]);
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
```

# 14. Codeforces Round #671 (Div. 2) D1. Sage's Birthday (easy version) 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1419/D1

Codeforces Round #671 (Div. 2) D1. Sage's Birthday (easy version) 

This is the easy version of the problem. The difference between the 
versions is that in the easy version all prices a_i are different. You can 
make hacks if and only if you solved both versions of the problem.

Today is Sage's birthday, and she will go shopping to buy ice spheres. All 
n ice spheres are placed in a row and they are numbered from 1 to n from 
left to right. Each ice sphere has a positive integer price. In this 
version all prices are different.

An ice sphere is cheap if it costs strictly less than two neighboring ice 
spheres: the nearest to the left and the nearest to the right. The 
leftmost and the rightmost ice spheres are not cheap. Sage will choose all 
cheap ice spheres and then buy only them.

You can visit the shop before Sage and reorder the ice spheres as you 
wish. Find out the maximum number of ice spheres that Sage can buy, and 
show how the ice spheres should be reordered.
*/
#define MAXN 100010
ll a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%lld",&a[i]);
    sort(a,a+n);
    int k=n/2;
    if(n%2)printf("%d\n",k);
    else printf("%d\n",k-1);
    for(int i=0;i<k;i++){
        printf("%lld %lld%c",a[k+i],a[i]," \n"[k+i==n-1]);
    }
    if(n%2)printf("%lld\n",a[n-1]);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 15. Educational Codeforces Round 96 (Rated for Div. 2) C. Numbers on Whiteboard 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1430/C

Educational Codeforces Round 96 (Rated for Div. 2) C. Numbers on Whiteboard 

Numbers 1, 2, 3, \dots n (each integer from 1 to n once) are written on a 
board. In one operation you can erase any two numbers a and b from the 
board and write one integer \frac{a + b}{2} rounded up instead.

You should perform the given operation n - 1 times and make the resulting 
number that will be left on the board as small as possible. 

For example, if n = 4, the following course of action is optimal:

 choose a = 4 and b = 2, so the new number is 3, and the whiteboard 
contains [1, 3, 3]; 

 choose a = 3 and b = 3, so the new number is 3, and the whiteboard 
contains [1, 3]; 

 choose a = 1 and b = 3, so the new number is 2, and the whiteboard 
contains [2]. 


It's easy to see that after n - 1 operations, there will be left only one 
number. Your goal is to minimize it.
*/
void run(){
    int n;scanf("%d",&n);
    int res=n;
    vector<string> ans;
    for(int i=n-1;i>=1;i--){
        ans.push_back(to_string(res)+" "+to_string(i));
        res=(res+i+1)/2;
    }
    printf("%d\n",res);
    for(const auto& s:ans)cout<<s<<endl;
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
```
