---
classes: wide2
title: "Codeforces Constructive Problems"
excerpt: "Practices of some codeforces constructive problems."
categories: 
  - coding
tags: 
  - constructive
toc: true
toc_sticky: true
mathjax: true
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

# 16. Codeforces Round #356 (Div. 2) B. Bear and Finding Criminals 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/680/B

Codeforces Round #356 (Div. 2) B. Bear and Finding Criminals 

There are n cities in Bearland, numbered 1 through n. Cities are arranged 
in one long row. The distance between cities i and j is equal to |i - j|.

Limak is a police officer. He lives in a city a. His job is to catch 
criminals. It's hard because he doesn't know in which cities criminals 
are. Though, he knows that there is at most one criminal in each city.

Limak is going to use a BCD (Bear Criminal Detector). The BCD will tell 
Limak how many criminals there are for every distance from a city a. After 
that, Limak can catch a criminal in each city for which he is sure that 
there must be a criminal.

You know in which cities criminals are. Count the number of criminals 
Limak will catch, after he uses the BCD.
*/
#define MAXN 200
int a[MAXN];
void run(){
    int n,b;scanf("%d%d",&n,&b);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    int res=0;
    for(int i=0;;i++){
        if(b+i>n&&b-i<1)break;
        if(b-i<1&&a[b+i]==1)res++;
        if(b+i>n&&a[b-i]==1)res++;
        if(b-i>=1&&b+i<=n&&a[b+i]==a[b-i])res+=a[b+i]+(i==0?0:a[b+i]);
    }
    printf("%d\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 17. Codeforces Round #668 (Div. 2) B. Array Cancellation 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1405/B

Codeforces Round #668 (Div. 2) B. Array Cancellation 

You're given an array a of n integers, such that a_1 + a_2 + \cdots + a_n 
= 0.

In one operation, you can choose two different indices i and j (1 \le i, j 
\le n), decrement a_i by one and increment a_j by one. If i &lt; j this 
operation is free, otherwise it costs one coin.

How many coins do you have to spend in order to make all elements equal to 
0?
*/
#define MAXN 100100
ll a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%lld",&a[i]);
    ll neg=0,res=0;
    for(int i=n;i>=1;i--){
        neg+=a[i];
        if(neg>0){
            res+=neg;
            neg=0;
        }
    }
    printf("%lld\n",res);
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

# 18. CodeCraft-22 and Codeforces Round #795 (Div. 2) B. Shoe Shuffling 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1691/B

CodeCraft-22 and Codeforces Round #795 (Div. 2) B. Shoe Shuffling 

A class of students got bored wearing the same pair of shoes every day, so 
they decided to shuffle their shoes among themselves. In this problem, a 
pair of shoes is inseparable and is considered as a single object.

There are n students in the class, and you are given an array s in 
non-decreasing order, where s_i is the shoe size of the i-th student. A 
shuffling of shoes is valid only if no student gets their own shoes and if 
every student gets shoes of size greater than or equal to their size. 

You have to output a permutation p of \{1,2,\ldots,n\} denoting a valid 
shuffling of shoes, where the i-th student gets the shoes of the p_i-th 
student (p_i \ne i). And output -1 if a valid shuffling does not exist.

A permutation is an array consisting of n distinct integers from 1 to n in 
arbitrary order. For example, [2,3,1,5,4] is a permutation, but [1,2,2] is 
not a permutation (2 appears twice in the array) and [1,3,4] is also not a 
permutation (n=3 but there is 4 in the array).
*/
#define MAXN 100010
ll a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=1;i<=n;i++)scanf("%lld",&a[i]);
    vector<ll> res;
    for(int i=1;i<=n;i++){
        int j=i;
        while(j<=n&&a[j]==a[i])j++;
        if(j==i+1){
            printf("-1\n");
            return;
        }
        res.push_back(j-1);
        for(int k=i;k<j-1;k++)res.push_back(k);
        i=j-1;
    }
    for(const auto& x:res)printf("%lld ",x);
    printf("\n");
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

# 19. Codeforces Round #744 (Div. 3) E1. Permutation Minimization by Deque 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1579/E1

Codeforces Round #744 (Div. 3) E1. Permutation Minimization by Deque 

In fact, the problems E1 and E2 do not have much in common. You should 
probably think of them as two separate problems.

A permutation p of size n is given. A permutation of size n is an array of 
size n in which each integer from 1 to n occurs exactly once. For example, 
[1, 4, 3, 2] and [4, 2, 1, 3] are correct permutations while [1, 2, 4] and 
[1, 2, 2] are not.

Let us consider an empty <a href="https://tinyurl.com/pfeucbux">deque

 (double-ended queue). A deque is a data structure that supports adding 
elements to both the beginning and the end. So, if there are elements [1, 
5, 2] currently in the deque, adding an element 4 to the beginning will 
produce the sequence [\color{red}{4}, 1, 5, 2], and adding same element to 
the end will produce [1, 5, 2, \color{red}{4}].

The elements of the permutation are sequentially added to the initially 
empty deque, starting with p_1 and finishing with p_n. Before adding each 
element to the deque, you may choose whether to add it to the beginning or 
the end.

For example, if we consider a permutation p = [3, 1, 2, 4], one of the 
possible sequences of actions looks like this: <table 
class="tex-tabular"><td class="tex-tabular-text-align-left">\quad 1.

<td class="tex-tabular-text-align-left">add 3 to the end of the deque:

<td class="tex-tabular-text-align-left">deque has a sequence 
[\color{red}{3}] in it;



<td class="tex-tabular-text-align-left">\quad 2.

<td class="tex-tabular-text-align-left">add 1 to the beginning of the 
deque:

<td class="tex-tabular-text-align-left">deque has a sequence 
[\color{red}{1}, 3] in it;



<td class="tex-tabular-text-align-left">\quad 3.

<td class="tex-tabular-text-align-left">add 2 to the end of the deque:

<td class="tex-tabular-text-align-left">deque has a sequence [1, 3, 
\color{red}{2}] in it;



<td class="tex-tabular-text-align-left">\quad 4.

<td class="tex-tabular-text-align-left">add 4 to the end of the deque:

<td class="tex-tabular-text-align-left">deque has a sequence [1, 3, 2, 
\color{red}{4}] in it;


Find the lexicographically smallest possible sequence of elements in the 
deque after the entire permutation has been processed. 

A sequence [x_1, x_2, \ldots, x_n] is lexicographically smaller than the 
sequence [y_1, y_2, \ldots, y_n] if there exists such i \leq n that x_1 = 
y_1, x_2 = y_2, \ldots, x_{i - 1} = y_{i - 1} and x_i &lt; y_i. In other 
words, if the sequences x and y have some (possibly empty) matching 
prefix, and the next element of the sequence x is strictly smaller than 
the corresponding element of the sequence y. For example, the sequence [1, 
3, 2, 4] is smaller than the sequence [1, 3, 4, 2] because after the two 
matching elements [1, 3] in the start the first sequence has an element 2 
which is smaller than the corresponding element 4 in the second sequence.
*/
#define MAXN 200202
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    deque<int> q;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(i==0)q.push_back(a[i]);
        else{
            if(a[i]<q.front())q.push_front(a[i]);
            else q.push_back(a[i]);
        }
    }
    for(const auto& x:q)printf("%d ",x);
    printf("\n");
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

# 20. Codeforces Round #812 (Div. 2) B. Optimal Reduction 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1713/B

Codeforces Round #812 (Div. 2) B. Optimal Reduction 

Consider an array a of n positive integers.

You may perform the following operation: 

 select two indices l and r (1 \leq l \leq r \leq n), then 

 decrease all elements a_l, a_{l + 1}, \dots, a_r by 1. 

Let's call f(a) the minimum number of operations needed to change array a 
into an array of n zeros.

Determine if for all permutations^\dagger b of a, f(a) \leq f(b) is true. 

^\dagger An array b is a permutation of an array a if b consists of the 
elements of a in arbitrary order. For example, [4,2,3,4] is a permutation 
of [3,2,4,4] while [1,2,2] is not a permutation of [1,2,3].
*/
#define MAXN 100101
ll a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%lld",&a[i]);
    }
    int pre=0;
    while(pre+1<n&&a[pre+1]>=a[pre])pre++;
    int suf=0;
    while(n-2-suf>=0&&a[n-2-suf]>=a[n-1-suf])suf++;
    if(pre+suf>=n-1)printf("YES\n");
    else printf("NO\n");
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


# 21. Codeforces Round #769 (Div. 2) B. Roof Construction 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1632/B

Codeforces Round #769 (Div. 2) B. Roof Construction 

It has finally been decided to build a roof over the football field in 
School 179. Its construction will require placing n consecutive vertical 
pillars. Furthermore, the headmaster wants the heights of all the pillars 
to form a permutation p of integers from 0 to n - 1, where p_i is the 
height of the i-th pillar from the left (1 \le i \le n).

As the chief, you know that the cost of construction of consecutive 
pillars is equal to the maximum value of the bitwise XOR of heights of all 
pairs of adjacent pillars. In other words, the cost of construction is 
equal to \max\limits_{1 \le i \le n - 1}{p_i \oplus p_{i + 1}}, where 
\oplus denotes the <a 
href="https://en.wikipedia.org/wiki/Bitwise_operation#XOR">bitwise XOR 
operation.

Find any sequence of pillar heights p of length n with the smallest 
construction cost.

In this problem, a permutation is an array consisting of n distinct 
integers from 0 to n - 1 in arbitrary order. For example, [2,3,1,0,4] is a 
permutation, but [1,0,1] is not a permutation (1 appears twice in the 
array) and [1,0,3] is also not a permutation (n=3, but 3 is in the array).
*/
void run(){
    int n;scanf("%d",&n);
    int lz=__builtin_clz(n-1);
    int mx=1<<(31-lz);
    for(int i=1;i<mx;i++)printf("%d ",i);
    printf("0 %d ",mx);
    for(int i=mx+1;i<n;i++)printf("%d ",i);
    printf("\n");
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

# 22. Codeforces Round #410 (Div. 2) A. Mike and palindrome 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/798/A

Codeforces Round #410 (Div. 2) A. Mike and palindrome 

Mike has a string s consisting of only lowercase English letters. He wants 
to change exactly one character from the string so that the resulting one 
is a palindrome. 

A palindrome is a string that reads the same backward as forward, for 
example strings "z", "aaa", "aba", "abccba" are palindromes, but strings 
"codeforces", "reality", "ab" are not.
*/
void run(){
    string s;cin>>s;
    int n=s.length();
    int cnt=0;
    for(int i=0;i<n/2;i++)if(s[i]!=s[n-1-i])cnt++;
    if(cnt==1||(cnt==0&&n%2))cout<<"YES"<<endl;
    else cout<<"NO"<<endl;
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

## 23. Codeforces Round #741 (Div. 2) B. Scenes From a Memory 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1562/B

Codeforces Round #741 (Div. 2) B. Scenes From a Memory 

During the hypnosis session, Nicholas suddenly remembered a positive 
integer n, which doesn't contain zeros in decimal notation. 

Soon, when he returned home, he got curious: what is the maximum number of 
digits that can be removed from the number so that the number becomes not 
prime, that is, either composite or equal to one?

For some numbers doing so is impossible: for example, for number 53 it's 
impossible to delete some of its digits to obtain a not prime integer. 
However, for all n in the test cases of this problem, it's guaranteed that 
it's possible to delete some of their digits to obtain a not prime number.

Note that you cannot remove all the digits from the number.

A prime number is a number that has no divisors except one and itself. A 
composite is a number that has more than two divisors. 1 is neither a 
prime nor a composite number.
*/
bool isp[200];
void init(){
    memset(isp,true,sizeof(isp));
    isp[0]=isp[1]=false;
    for(int i=2;i<200;i++)if(isp[i])for(int j=i*i;j<200;j+=i)isp[j]=false;
}
void run(){
    int n;scanf("%d",&n);
    string s;cin>>s;
    vector<int> a;
    int pcnt=0;
    for(int i=0;i<n;i++){
        a.push_back(s[i]-'0');
        if(!isp[a[i]]){
            printf("1\n%d\n",a[i]);
            return;
        }
    }
    for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++){
            int x=a[i]*10+a[j];
            if(!isp[x]){
                printf("2\n%d\n",x);
                return;
            }
        }
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

## 24. Codeforces Round #839 (Div. 3) C. Different Differences 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1772/C

Codeforces Round #839 (Div. 3) C. Different Differences 

An array a consisting of k integers is strictly increasing if a_1 &lt; a_2 
&lt; \dots &lt; a_k. For example, the arrays [1, 3, 5], [1, 2, 3, 4], [3, 
5, 6] are strictly increasing; the arrays [2, 2], [3, 7, 5], [7, 4, 3], 
[1, 2, 2, 3] are not.

For a strictly increasing array a of k elements, let's denote the 
characteristic as the number of different elements in the array [a_2 - 
a_1, a_3 - a_2, \dots, a_k - a_{k-1}]. For example, the characteristic of 
the array [1, 3, 4, 7, 8] is 3 since the array [2, 1, 3, 1] contains 3 
different elements: 2, 1 and 3.

You are given two integers k and n (k \le n). Construct an increasing 
array of k integers from 1 to n with maximum possible characteristic.
*/
void run(){
    int k,n;scanf("%d%d",&k,&n);
    int x=1,cnt=0;
    int d=1;
    while(true){
        if(n-x>=k-cnt-1){
            printf("%d ",x);
        }else{
            for(int i=cnt;i<k;i++)printf("%d ",x-d+2+i-cnt);
            printf("\n");
            return;
        }
        cnt++;
        if(cnt>=k)break;
        x+=d;d++;
    }
    printf("\n");
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
