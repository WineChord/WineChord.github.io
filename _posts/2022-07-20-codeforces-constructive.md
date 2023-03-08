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

# 23. Codeforces Round #741 (Div. 2) B. Scenes From a Memory 

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

# 24. Codeforces Round #839 (Div. 3) C. Different Differences 

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

# 25. Codeforces Round #816 (Div. 2) B. Beautiful Array 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1715/B

Codeforces Round #816 (Div. 2) B. Beautiful Array 

Stanley defines the beauty of an array a of length n, which contains 
non-negative integers, as follows: \sum\limits_{i = 1}^{n} \left \lfloor 
\frac{a_{i}}{k} \right \rfloor, which means that we divide each element by 
k, round it down, and sum up the resulting values.

Stanley told Sam the integer k and asked him to find an array a of n 
non-negative integers, such that the beauty is equal to b and the sum of 
elements is equal to s. Help Sam — find any of the arrays satisfying the 
conditions above.
*/
void run(){
    ll n,k,b,s;
    scanf("%lld%lld%lld%lld",&n,&k,&b,&s);
    // [kb,kb+n*(k-1)]
    ll mi=k*b;
    ll mx=mi+n*(k-1);
    if(s<mi||s>mx){
        printf("-1\n");
        return;
    }
    ll r=s-mi;
    ll base=r/n;
    ll rem=r%n;
    for(int i=0;i<rem;i++)printf("%lld ",base+1);
    for(int i=rem;i<n-1;i++)printf("%lld ",base);
    printf("%lld\n",base+mi);
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

# 26. Codeforces Round #682 (Div. 2) B. Valerii Against Everyone 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1438/B

Codeforces Round #682 (Div. 2) B. Valerii Against Everyone 

You're given an array b of length n. Let's define another array a, also of 
length n, for which a_i = 2^{b_i} (1 \leq i \leq n). 

Valerii says that every two non-intersecting subarrays of a have different 
sums of elements. You want to determine if he is wrong. More formally, you 
need to determine if there exist four integers l_1,r_1,l_2,r_2 that 
satisfy the following conditions: 

 1 \leq l_1 \leq r_1 \lt l_2 \leq r_2 \leq n; 

 a_{l_1}+a_{l_1+1}+\ldots+a_{r_1-1}+a_{r_1} = 
a_{l_2}+a_{l_2+1}+\ldots+a_{r_2-1}+a_{r_2}. 

If such four integers exist, you will prove Valerii wrong. Do they exist?

An array c is a subarray of an array d if c can be obtained from d by 
deletion of several (possibly, zero or all) elements from the beginning 
and several (possibly, zero or all) elements from the end.
*/
#define MAXN 1101
ll b[MAXN];
void run(){
    int n;scanf("%d",&n);
    unordered_map<ll,int> mp;
    bool res=false;
    for(int i=0;i<n;i++){
        scanf("%lld",&b[i]);
        if(mp.find(b[i])!=mp.end()){
            res=true;
        }
        mp[b[i]]=1;
    }
    if(res)printf("YES\n");
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

# 27. Codeforces Round #770 (Div. 2) C. OKEA 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1634/C

Codeforces Round #770 (Div. 2) C. OKEA 

<div class="epigraph"><div class="epigraph-text">People worry that 
computers will get too smart and take over the world, but the real problem 
is that they're too stupid and they've already taken over the world.

<div class="epigraph-source">— Pedro Domingos

You work for a well-known department store that uses leading technologies 
and employs mechanistic work — that is, robots!

The department you work in sells n \cdot k items. The first item costs 1 
dollar, the second item costs 2 dollars, and so on: i-th item costs i 
dollars. The items are situated on shelves. The items form a rectangular 
grid: there are n shelves in total, and each shelf contains exactly k 
items. We will denote by a_{i,j} the price of j-th item (counting from the 
left) on the i-th shelf, 1 \le i \le n, 1 \le j \le k.

Occasionally robots get curious and ponder on the following question: what 
is the mean price (arithmetic average) of items a_{i,l}, a_{i,l+1}, 
\ldots, a_{i,r} for some shelf i and indices l \le r? Unfortunately, the 
old robots can only work with whole numbers. If the mean price turns out 
not to be an integer, they break down.

You care about robots' welfare. You want to arrange the items in such a 
way that the robots cannot theoretically break. Formally, you want to 
choose such a two-dimensional array a that:

 Every number from 1 to n \cdot k (inclusively) occurs exactly once. 

 For each i, l, r, the mean price of items from l to r on i-th shelf is an 
integer. 

Find out if such an arrangement is possible, and if it is, give any 
example of such arrangement.
*/
void run(){
    int n,k;scanf("%d%d",&n,&k);
    if(k==1){
        printf("YES\n");
        for(int i=1;i<=n;i++)printf("%d\n",i);
        return;
    }
    if(n%2){
        printf("NO\n");
        return;
    }
    printf("YES\n");
    for(int i=1;i<=n;i++){
        for(int j=0;j<k;j++){
            printf("%d%c",i+j*n," \n"[j==k-1]);
        }
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

# 28. Good Bye 2022: 2023 is NEAR B. Koxia and Permutation 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1770/B

Good Bye 2022: 2023 is NEAR B. Koxia and Permutation 

Reve has two integers n and k.

Let p be a permutation^\dagger of length n. Let c be an array of length n 
- k + 1 such that c_i = \max(p_i, \dots, p_{i+k-1}) + \min(p_i, \dots, 
p_{i+k-1}). Let the cost of the permutation p be the maximum element of c.

Koxia wants you to construct a permutation with the minimum possible cost.

^\dagger A permutation of length n is an array consisting of n distinct 
integers from 1 to n in arbitrary order. For example, [2,3,1,5,4] is a 
permutation, but [1,2,2] is not a permutation (2 appears twice in the 
array), and [1,3,4] is also not a permutation (n=3 but there is 4 in the 
array).
*/
void run(){
    int n,k;scanf("%d%d",&n,&k);
    int l=1,r=n;bool flag=true;
    while(l<=r){
        printf("%d ",flag==true?r--:l++);
        flag=!flag;
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

# 29. Codeforces Round #756 (Div. 3) C. Polycarp Recovers the Permutation 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1611/C

Codeforces Round #756 (Div. 3) C. Polycarp Recovers the Permutation 

Polycarp wrote on a whiteboard an array p of length n, which is a 
permutation of numbers from 1 to n. In other words, in p each number from 
1 to n occurs exactly once.

He also prepared a resulting array a, which is initially empty (that is, 
it has a length of 0).

After that, he did exactly n steps. Each step looked like this:

 Look at the leftmost and rightmost elements of p, and pick the smaller of 
the two.

 If you picked the leftmost element of p, append it to the left of a; 
otherwise, if you picked the rightmost element of p, append it to the 
right of a.

 The picked element is erased from p. 

Note that on the last step, p has a length of 1 and its minimum element is 
both leftmost and rightmost. In this case, Polycarp can choose what role 
the minimum element plays. In other words, this element can be added to a 
both on the left and on the right (at the discretion of Polycarp).

Let's look at an example. Let n=4, p=[3, 1, 4, 2]. Initially a=[]. Then:

 During the first step, the minimum is on the right (with a value of 2), 
so after this step, p=[3,1,4] and a=[2] (he added the value 2 to the 
right). 

 During the second step, the minimum is on the left (with a value of 3), 
so after this step, p=[1,4] and a=[3,2] (he added the value 3 to the 
left). 

 During the third step, the minimum is on the left (with a value of 1), so 
after this step, p=[4] and a=[1,3,2] (he added the value 1 to the left). 

 During the fourth step, the minimum is both left and right (this value is 
4). Let's say Polycarp chose the right option. After this step, p=[] and 
a=[1,3,2,4] (he added the value 4 to the right).

Thus, a possible value of a after n steps could be a=[1,3,2,4].

You are given the final value of the resulting array a. Find any possible 
initial value for p that can result the given a, or determine that there 
is no solution.
*/
#define MAXN 200200
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    if(a[0]!=n&&a[n-1]!=n){
        printf("-1\n");
        return;
    }
    for(int i=n-1;i>=0;i--)printf("%d%c",a[i]," \n"[i==0]);
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

# 30. Codeforces Round #757 (Div. 2) B. Divan and a New Project  

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1614/B

Codeforces Round #757 (Div. 2) B. Divan and a New Project  

The company "Divan's Sofas" is planning to build n + 1 different buildings 
on a coordinate line so that: 

 the coordinate of each building is an integer number; 

 no two buildings stand at the same point. 

Let x_i be the coordinate of the i-th building. To get from the building i 
to the building j, Divan spends |x_i - x_j| minutes, where |y| is the 
absolute value of y.

All buildings that Divan is going to build can be numbered from 0 to n. 
The businessman will live in the building 0, the new headquarters of 
"Divan's Sofas". In the first ten years after construction Divan will 
visit the i-th building a_i times, each time spending 2 \cdot |x_0-x_i| 
minutes for walking.

Divan asks you to choose the coordinates for all n + 1 buildings so that 
over the next ten years the businessman will spend as little time for 
walking as possible.
*/
void run(){
    int n;scanf("%d",&n);
    vector<pair<int,int>> a;
    for(int i=1;i<=n;i++){
        int x;scanf("%d",&x);
        a.push_back({-x,i});
    }
    sort(a.begin(),a.end());
    vector<int> res(n+1,0);
    int cnt=1;
    ll ans=0;
    for(int i=0;i<n;i++){
        res[a[i].second]=cnt;
        ans+=2ll*abs(cnt)*(-a[i].first);
        if(cnt<0)cnt=-cnt+1;
        else cnt=-cnt;
    }
    printf("%lld\n",ans);
    for(int i=0;i<=n;i++)printf("%d%c",res[i]," \n"[i==n]);
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

# 31. Codeforces Global Round 16 C. MAX-MEX Cut 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1566/C

Codeforces Global Round 16 C. MAX-MEX Cut 

A binary string is a string that consists of characters 0 and 1. A 
bi-table is a table that has exactly two rows of equal length, each being 
a binary string.

Let \operatorname{MEX} of a bi-table be the smallest digit among 0, 1, or 
2 that does not occur in the bi-table. For example, \operatorname{MEX} for 
\begin{bmatrix} 0011\\ 1010 \end{bmatrix} is 2, because 0 and 1 occur in 
the bi-table at least once. \operatorname{MEX} for \begin{bmatrix} 111\\ 
111 \end{bmatrix} is 0, because 0 and 2 do not occur in the bi-table, and 
0 &lt; 2.

You are given a bi-table with n columns. You should cut it into any number 
of bi-tables (each consisting of consecutive columns) so that each column 
is in exactly one bi-table. It is possible to cut the bi-table into a 
single bi-table — the whole bi-table.

What is the maximal sum of \operatorname{MEX} of all resulting bi-tables 
can be?
*/
int solve(const string& s){
    int res=count(s.begin(),s.end(),'0');
    bool x=false,y=false;
    for(const auto& c:s){
        if(c=='0')x=true;
        if(c=='1')y=true;
        if(x&&y){
            res++;
            x=false;y=false;
        }
    }
    return res;
}
void run(){
    int n;cin>>n;
    string s,t;cin>>s>>t;
    int res=0;
    string k;
    for(int i=0;i<n;i++){
        if(s[i]!=t[i]){
            res+=2;
            res+=solve(k);
            k="";
        }else k+=s[i];
    }
    cout<<res+solve(k)<<"\n";
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    cin.tie(0);cout.tie(0);
    ios_base::sync_with_stdio(false);
    int T;cin>>T;
    while(T--){
        run();
    }
}
```

# 32. Codeforces Round #563 (Div. 2) A. Ehab Fails to Be Thanos 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1174/A

Codeforces Round #563 (Div. 2) A. Ehab Fails to Be Thanos 

You're given an array a of length 2n. Is it possible to reorder it in such 
way so that the sum of the first n elements isn't equal to the sum of the 
last n elements?
*/
#define MAXN 3030
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<2*n;i++)scanf("%d",&a[i]);
    sort(a,a+2*n);
    if(accumulate(a,a+n,0)==accumulate(a+n,a+2*n,0)){printf("-1\n");return;}
    for(int i=0;i<2*n;i++)printf("%d%c",a[i]," \n"[i==2*n-1]);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 33. Codeforces Round #352 (Div. 2) B. Different is Good 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/672/B

Codeforces Round #352 (Div. 2) B. Different is Good 

A wise man told Kerem "Different is good" once, so Kerem wants all things 
in his life to be different. 

Kerem recently got a string s consisting of lowercase English letters. 
Since Kerem likes it when things are different, he wants all substrings of 
his string s to be distinct. Substring is a string formed by some number 
of consecutive characters of the string. For example, string "aba" has 
substrings "" (empty substring), "a", "b", "a", "ab", "ba", "aba".

If string s has at least two equal substrings then Kerem will change 
characters at some positions to some other lowercase English letters. 
Changing characters is a very tiring job, so Kerem want to perform as few 
changes as possible.

Your task is to find the minimum number of changes needed to make all the 
substrings of the given string distinct, or determine that it is 
impossible.
*/
void run(){
    int n;string s;
    cin>>n>>s;
    unordered_map<char,int> mp;
    for(const auto& c:s)mp[c]++;
    int res=0;
    for(const auto&[k,v]:mp)res+=v-1;
    if(mp.size()+res>26){
        printf("-1\n");
        return;
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

# 34. Codeforces Round #782 (Div. 2) A. Red Versus Blue 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1659/A

Codeforces Round #782 (Div. 2) A. Red Versus Blue 

Team Red and Team Blue competed in a competitive FPS. Their match was 
streamed around the world. They played a series of n matches.

In the end, it turned out Team Red won r times and Team Blue won b times. 
Team Blue was less skilled than Team Red, so b was strictly less than r.

You missed the stream since you overslept, but you think that the match 
must have been neck and neck since so many people watched it. So you 
imagine a string of length n where the i-th character denotes who won the 
i-th match  — it is R if Team Red won or B if Team Blue won. You imagine 
the string was such that the maximum number of times a team won in a row 
was as small as possible. For example, in the series of matches RBBRRRB, 
Team Red won 3 times in a row, which is the maximum.

You must find a string satisfying the above conditions. If there are 
multiple answers, print any.
*/
void run(){
    int n,r,b;scanf("%d%d%d",&n,&r,&b);
    int base=r/(b+1);
    int rem=r%(b+1);
    for(int i=0;i<b+1;i++){
        if(i<rem)printf("R");
        for(int j=0;j<base;j++)printf("R");
        if(i<b)printf("B");
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

# 35. Codeforces Round #411 (Div. 2) B. 3-palindrome 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/805/B

Codeforces Round #411 (Div. 2) B. 3-palindrome 

In the beginning of the new year Keivan decided to reverse his name. He 
doesn't like palindromes, so he changed Naviek to Navick.

He is too selfish, so for a given n he wants to obtain a string of n 
characters, each of which is either 'a', 'b' or 'c', with no palindromes 
of length 3 appearing in the string as a substring. For example, the 
strings "abc" and "abca" suit him, while the string "aba" doesn't. He also 
want the number of letters 'c' in his string to be as little as possible.
*/
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        int k=i%4;
        if(k<2)printf("b");
        else printf("a");
    }
    printf("\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 36. Codeforces Round #411 (Div. 1) A. Find Amir 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/804/A

Codeforces Round #411 (Div. 1) A. Find Amir 

A few years ago Sajjad left his school and register to another one due to 
security reasons. Now he wishes to find Amir, one of his schoolmates and 
good friends.

There are n schools numerated from 1 to n. One can travel between each 
pair of them, to do so, he needs to buy a ticket. The ticker between 
schools i and j costs <img align="middle" class="tex-formula" 
src="https://espresso.codeforces.com/48a4e5770c4d95918ab20e59965b1361f52acc
70.png" style="max-width: 100.0%;max-height: 100.0%;" /> and can be used 
multiple times. Help Sajjad to find the minimum cost he needs to pay for 
tickets to visit all schools. He can start and finish in any school.
*/
void run(){
    int n;scanf("%d",&n);
    printf("%d\n",(n-1)/2);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 37. Codeforces Round #604 (Div. 2) A. Beautiful String 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1265/A

Codeforces Round #604 (Div. 2) A. Beautiful String 

A string is called beautiful if no two consecutive characters are equal. 
For example, "ababcb", "a" and "abab" are beautiful strings, while 
"aaaaaa", "abaa" and "bb" are not.

Ahcl wants to construct a beautiful string. He has a string s, consisting 
of only characters 'a', 'b', 'c' and '?'. Ahcl needs to replace each 
character '?' with one of the three characters 'a', 'b' or 'c', such that 
the resulting string is beautiful. Please help him!

More formally, after replacing all characters '?', the condition s_i \neq 
s_{i+1} should be satisfied for all 1 \leq i \leq |s| - 1, where |s| is 
the length of the string s.
*/
void run(){
    string s;cin>>s;
    int n=s.length();
    for(int i=0;i<n-1;i++){
        if(s[i]!='?'&&s[i]==s[i+1]){
            printf("-1\n");
            return;
        }
    }
    int pre=0;
    for(int i=0;i<n;i++){
        if(s[i]!='?'){
            pre=s[i]-'a';
            continue;
        }
        pre=(pre+1)%3;
        s[i]='a'+pre;
        if(i+1<n&&s[i+1]==s[i]){
            pre=(pre+1)%3;
            s[i]='a'+pre;
        }
    }
    cout<<s<<"\n";
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

# 38. Codeforces Round #550 (Div. 3) C. Two Shuffled Sequences 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1144/C

Codeforces Round #550 (Div. 3) C. Two Shuffled Sequences 

Two integer sequences existed initially — one of them was strictly 
increasing, and the other one — strictly decreasing.

Strictly increasing sequence is a sequence of integers [x_1 &lt; x_2 &lt; 
\dots &lt; x_k]. And strictly decreasing sequence is a sequence of 
integers [y_1 > y_2 > \dots > y_l]. Note that the empty sequence and the 
sequence consisting of one element can be considered as increasing or 
decreasing.

They were merged into one sequence a. After that sequence a got shuffled. 
For example, some of the possible resulting sequences a for an increasing 
sequence [1, 3, 4] and a decreasing sequence [10, 4, 2] are sequences [1, 
2, 3, 4, 4, 10] or [4, 2, 1, 10, 4, 3].

This shuffled sequence a is given in the input.

Your task is to find any two suitable initial sequences. One of them 
should be strictly increasing and the other one — strictly decreasing. 
Note that the empty sequence and the sequence consisting of one element 
can be considered as increasing or decreasing.

If there is a contradiction in the input and it is impossible to split the 
given sequence a to increasing and decreasing sequences, print "NO".
*/
#define MAXN 200202
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    unordered_map<int,int> mp;
    bool flag=false;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        mp[a[i]]++;
        if(mp[a[i]]>=3)flag=true;
    }
    if(flag){
        printf("NO\n");
        return;
    }
    sort(a,a+n);
    vector<int> x,y;
    for(int i=0;i<n;i++){
        if(i==0)x.push_back(a[i]);
        else{
            if(a[i]==a[i-1])y.push_back(a[i]);
            else x.push_back(a[i]);
        }
    }
    if(y.empty()){
        y.push_back(x.back());
        x.pop_back();
    }
    reverse(y.begin(),y.end());
    printf("YES\n");
    printf("%d\n",x.size());
    for(const auto&v:x)printf("%d ",v);
    printf("\n");
    printf("%d\n",y.size());
    for(const auto&v:y)printf("%d ",v);
    printf("\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 39. Codeforces Round #394 (Div. 2) A. Dasha and Stairs 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/761/A

Codeforces Round #394 (Div. 2) A. Dasha and Stairs 

On her way to programming school tiger Dasha faced her first test — a huge 
staircase!

 <img class="tex-graphics" 
src="https://espresso.codeforces.com/31ae258b18250cd0831c18db6108441e9b6731
17.png" style="max-width: 100.0%;max-height: 100.0%;" /> 

The steps were numbered from one to infinity. As we know, tigers are very 
fond of all striped things, it is possible that it has something to do 
with their color. So on some interval of her way she calculated two values 
— the number of steps with even and odd numbers. 

You need to check whether there is an interval of steps from the l-th to 
the r-th (1 ≤ l ≤ r), for which values that Dasha has found are correct.
*/
void run(){
    int a,b;scanf("%d%d",&a,&b);
    if(a==b&&a==0){
        printf("NO\n");
        return;
    }
    if(abs(a-b)<2)printf("YES\n");
    else printf("NO\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 40. Codeforces Global Round 6 B. Dice Tower 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1266/B

Codeforces Global Round 6 B. Dice Tower 

Bob is playing with 6-sided dice. A net of such standard cube is shown 
below.

<img class="tex-graphics" 
src="https://espresso.codeforces.com/0065ec3465bffc3d36eefbdb4a5cab1e259814
6d.png" style="max-width: 100.0%;max-height: 100.0%;" />

He has an unlimited supply of these dice and wants to build a tower by 
stacking multiple dice on top of each other, while choosing the 
orientation of each dice. Then he counts the number of visible pips on the 
faces of the dice.

For example, the number of visible pips on the tower below is 29 — the 
number visible on the top is 1, from the south 5 and 3, from the west 4 
and 2, from the north 2 and 4 and from the east 3 and 5.

<img class="tex-graphics" 
src="https://espresso.codeforces.com/8b7b0bd353a0dc7abd2f410cbb90999b99b2a2
a7.png" style="max-width: 100.0%;max-height: 100.0%;" />

The one at the bottom and the two sixes by which the dice are touching are 
not visible, so they are not counted towards total.

Bob also has t favourite integers x_i, and for every such integer his goal 
is to build such a tower that the number of visible pips is exactly x_i. 
For each of Bob's favourite integers determine whether it is possible to 
build a tower that has exactly that many visible pips.
*/
void run(){
    ll x;scanf("%lld",&x);
    ll y=x/14;
    ll r=x%14;
    if(y>=1&&(r>=1&&r<=6))printf("YES\n");
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

# 41. Codeforces Round #830 (Div. 2) A. Bestie 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1732/A

Codeforces Round #830 (Div. 2) A. Bestie 

You are given an array a consisting of n integers a_1, a_2, \ldots, a_n. 
Friends asked you to make the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor (GCD)

 of all numbers in the array equal to 1. In one operation, you can do the 
following:

 Select an arbitrary index in the array 1 \leq i \leq n;

 Make a_i = \gcd(a_i, i), where \gcd(x, y) denotes the GCD of integers x 
and y. The cost of such an operation is n - i + 1.

You need to find the minimum total cost of operations we need to perform 
so that the GCD of the all array numbers becomes equal to 1.
*/
unordered_set<ll> mp={2,3,5,7,11,13,17,19};
void run(){
    int n;scanf("%d",&n);
    int res=INT_MAX;
    int tot=-1;
    vector<int> can;
    for(int i=1;i<=n;i++){
        int x;scanf("%d",&x);
        if(tot==-1)tot=x;
        else tot=__gcd(tot,x);
    }
    if(tot==1)res=0;
    else if(__gcd(tot,n)==1)res=1;
    else if(__gcd(tot,n-1)==1)res=2;
    else res=3;
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

# 42. Codeforces Round #546 (Div. 2) B. Nastya Is Playing Computer Games 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1136/B

Codeforces Round #546 (Div. 2) B. Nastya Is Playing Computer Games 

Finished her homework, Nastya decided to play computer games. Passing 
levels one by one, Nastya eventually faced a problem. Her mission is to 
leave a room, where a lot of monsters live, as quickly as possible.

There are n manholes in the room which are situated on one line, but, 
unfortunately, all the manholes are closed, and there is one stone on 
every manhole. There is exactly one coin under every manhole, and to win 
the game Nastya should pick all the coins. Initially Nastya stands near 
the k-th manhole from the left. She is thinking what to do.

In one turn, Nastya can do one of the following: 

 if there is at least one stone on the manhole Nastya stands near, throw 
exactly one stone from it onto any other manhole (yes, Nastya is strong). 

 go to a neighboring manhole; 

 if there are no stones on the manhole Nastya stays near, she can open it 
and pick the coin from it. After it she must close the manhole immediately 
(it doesn't require additional moves). 



 <img class="tex-graphics" 
src="https://espresso.codeforces.com/d67a584d94f164baaa23c9f6ada7b629599b01
b9.png" style="max-width: 100.0%;max-height: 100.0%;" />   The figure 
shows the intermediate state of the game. At the current position Nastya 
can throw the stone to any other manhole or move left or right to the 
neighboring manholes. If she were near the leftmost manhole, she could 
open it (since there are no stones on it). 

Nastya can leave the room when she picks all the coins. Monsters are 
everywhere, so you need to compute the minimum number of moves Nastya has 
to make to pick all the coins.

Note one time more that Nastya can open a manhole only when there are no 
stones onto it.
*/
void run(){
    int n,k;scanf("%d%d",&n,&k);
    printf("%d\n",6+3*(n-2)+min(n-k,k-1));
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 43. Pinely Round 1 (Div. 1 + Div. 2) B. Elimination of a Ring 


```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1761/B

Pinely Round 1 (Div. 1 + Div. 2) B. Elimination of a Ring 

Define a cyclic sequence of size n as an array s of length n, in which s_n 
is adjacent to s_1.

Muxii has a ring represented by a cyclic sequence a of size n.

However, the ring itself hates equal adjacent elements. So if two adjacent 
elements in the sequence are equal at any time, one of them will be erased 
immediately. The sequence doesn't contain equal adjacent elements 
initially.

Muxii can perform the following operation until the sequence becomes empty:

 Choose an element in a and erase it. 

For example, if ring is [1, 2, 4, 2, 3, 2], and Muxii erases element 4, 
then ring would erase one of the elements equal to 2, and the ring will 
become [1, 2, 3, 2].

Muxii wants to find the maximum number of operations he could perform.

Note that in a ring of size 1, its only element isn't considered adjacent 
to itself (so it's not immediately erased).
*/
void run(){
    int n;scanf("%d",&n);
    unordered_map<int,int> mp;
    for(int i=0;i<n;i++){
        int x;
        scanf("%d",&x);
        mp[x]++;

    }
    if(mp.size()==2)printf("%d\n",n/2+1);
    else printf("%d\n",n);
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

# 44. Codeforces Round #401 (Div. 2) A. Shell Game 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/777/A

Codeforces Round #401 (Div. 2) A. Shell Game 

Bomboslav likes to look out of the window in his room and watch lads 
outside playing famous shell game. The game is played by two persons: 
operator and player. Operator takes three similar opaque shells and places 
a ball beneath one of them. Then he shuffles the shells by swapping some 
pairs and the player has to guess the current position of the ball.

Bomboslav noticed that guys are not very inventive, so the operator always 
swaps the left shell with the middle one during odd moves (first, third, 
fifth, etc.) and always swaps the middle shell with the right one during 
even moves (second, fourth, etc.).

Let's number shells from 0 to 2 from left to right. Thus the left shell is 
assigned number 0, the middle shell is 1 and the right shell is 2. 
Bomboslav has missed the moment when the ball was placed beneath the 
shell, but he knows that exactly n movements were made by the operator and 
the ball was under shell x at the end. Now he wonders, what was the 
initial position of the ball?
*/
int a[][3]={
    {0,1,2},
    {1,0,2},
    {1,2,0},
    {2,1,0},
    {2,0,1},
    {0,2,1},
    {0,1,2},
};
void run(){
    int n,x;scanf("%d%d",&n,&x);
    printf("%d\n",a[n%6][x]);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 45. Codeforces Round #493 (Div. 2) A. Balloons 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/998/A

Codeforces Round #493 (Div. 2) A. Balloons 

There are quite a lot of ways to have fun with inflatable balloons. For 
example, you can fill them with water and see what happens.

Grigory and Andrew have the same opinion. So, once upon a time, they went 
to the shop and bought n packets with inflatable balloons, where i-th of 
them has exactly a_i balloons inside.

They want to divide the balloons among themselves. In addition, there are 
several conditions to hold:

 Do not rip the packets (both Grigory and Andrew should get unbroken 
packets); 

 Distribute all packets (every packet should be given to someone); 

 Give both Grigory and Andrew at least one packet; 

 To provide more fun, the total number of balloons in Grigory's packets 
should not be equal to the total number of balloons in Andrew's packets. 

Help them to divide the balloons or determine that it's impossible under 
these conditions.
*/
#define MAXN 20
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    int mi=INT_MAX,idx=-1;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(a[i]<mi){
            mi=a[i];
            idx=i;
        }
    }
    if(n==1||(n==2&&a[0]==a[1])){
        printf("-1\n");
        return;
    }
    printf("1\n%d\n",idx+1);

}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 46. Codeforces Round #402 (Div. 2) A. Pupils Redistribution 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/779/A

Codeforces Round #402 (Div. 2) A. Pupils Redistribution 

In Berland each high school student is characterized by academic 
performance — integer value between 1 and 5.

In high school 0xFF there are two groups of pupils: the group A and the 
group B. Each group consists of exactly n students. An academic 
performance of each student is known — integer value between 1 and 5.

The school director wants to redistribute students between groups so that 
each of the two groups has the same number of students whose academic 
performance is equal to 1, the same number of students whose academic 
performance is 2 and so on. In other words, the purpose of the school 
director is to change the composition of groups, so that for each value of 
academic performance the numbers of students in both groups are equal.

To achieve this, there is a plan to produce a series of exchanges of 
students between groups. During the single exchange the director selects 
one student from the class A and one student of class B. After that, they 
both change their groups.

Print the least number of exchanges, in order to achieve the desired equal 
numbers of students for each academic performance.
*/
#define MAXN 110
int a[MAXN],b[MAXN];
void run(){
    int n;scanf("%d",&n);
    unordered_map<int,vector<int>> mp1;
    unordered_map<int,vector<int>> mp2;
    unordered_map<int,int> exp;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        mp1[a[i]].push_back(i);
        exp[a[i]]++;
    }
    for(int i=0;i<n;i++){
        scanf("%d",&b[i]);
        mp2[b[i]].push_back(i);
        exp[b[i]]++;
    }
    bool flag=false;
    for(const auto&[k,v]:exp){
        if(v%2){
            flag=true;
            break;
        }
        exp[k]=v/2;
    }
    if(flag){
        printf("-1\n");
        return;
    }
    int res=0;
    for(const auto&[k,v]:mp1){
        int sz=v.size();
        if(exp[k]<sz)res+=sz-exp[k];
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

# 47. Codeforces Round #209 (Div. 2) A. Table 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/359/A

Codeforces Round #209 (Div. 2) A. Table 

Simon has a rectangular table consisting of n rows and m columns. Simon 
numbered the rows of the table from top to bottom starting from one and 
the columns — from left to right starting from one. We'll represent the 
cell on the x-th row and the y-th column as a pair of numbers (x, y). The 
table corners are cells: (1, 1), (n, 1), (1, m), (n, m).

Simon thinks that some cells in this table are good. Besides, it's known 
that no good cell is the corner of the table. 

Initially, all cells of the table are colorless. Simon wants to color all 
cells of his table. In one move, he can choose any good cell of table 
(x_1, y_1), an arbitrary corner of the table (x_2, y_2) and color all 
cells of the table (p, q), which meet both inequations: min(x_1, x_2) ≤ p 
≤ max(x_1, x_2), min(y_1, y_2) ≤ q ≤ max(y_1, y_2).

Help Simon! Find the minimum number of operations needed to color all 
cells of the table. Note that you can color one cell multiple times.
*/
#define MAXN 100
int a[MAXN][MAXN];
void run(){
    int n,m;scanf("%d%d",&n,&m);
    bool flag=false;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            scanf("%d",&a[i][j]);
            if(a[i][j]){
                if(i==0||i==n-1||j==0||j==m-1)flag=true;
            }
        }
    }
    if(flag)printf("2\n");
    else printf("4\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    // int T;scanf("%d",&T);
    // while(T--){
        run();
    // }
}
```

# 48. VK Cup 2016 - Qualification Round 1 A. Voting for Photos 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/637/A

VK Cup 2016 - Qualification Round 1 A. Voting for Photos 

After celebrating the midcourse the students of one of the faculties of 
the Berland State University decided to conduct a vote for the best photo. 
They published the photos in the social network and agreed on the rules to 
choose a winner: the photo which gets most likes wins. If multiple photoes 
get most likes, the winner is the photo that gets this number first.

Help guys determine the winner photo by the records of likes.
*/
void run(){
    int n;scanf("%d",&n);
    int mx=0,idx=-1;
    unordered_map<int,int> mp;
    for(int i=1;i<=n;i++){
        int x;scanf("%d",&x);
        mp[x]++;
        if(mp[x]>mx){
            mx=mp[x];
            idx=x;
        }
    }
    printf("%d\n",idx);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 49. CROC 2016 - Qualification A. Parliament of Berland 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/644/A

CROC 2016 - Qualification A. Parliament of Berland 

There are n parliamentarians in Berland. They are numbered with integers 
from 1 to n. It happened that all parliamentarians with odd indices are 
Democrats and all parliamentarians with even indices are Republicans.

New parliament assembly hall is a rectangle consisting of a × b chairs — a 
rows of b chairs each. Two chairs are considered neighbouring if they 
share as side. For example, chair number 5 in row number 2 is neighbouring 
to chairs number 4 and 6 in this row and chairs with number 5 in rows 1 
and 3. Thus, chairs have four neighbours in general, except for the chairs 
on the border of the hall

We know that if two parliamentarians from one political party (that is two 
Democrats or two Republicans) seat nearby they spent all time discussing 
internal party issues.

Write the program that given the number of parliamentarians and the sizes 
of the hall determine if there is a way to find a seat for any 
parliamentarian, such that no two members of the same party share 
neighbouring seats.
*/
void run(){
    int n,a,b;scanf("%d%d%d",&n,&a,&b);
    if(a*b<n){
        printf("-1\n");
        return;
    }
    vector<vector<int>> v(a+1,vector<int>(b+1,0));
    int odd=1,even=2,cnt=0;;
    for(int i=1;i<=a;i++){
        for(int j=1;j<=b;j++){
            if(cnt<n){
                if(i%2==j%2){
                    if(odd<=n){
                        v[i][j]=odd;
                        odd+=2;
                        cnt++;
                    }
                }else{
                    if(even<=n){
                        v[i][j]=even;
                        even+=2;
                        cnt++;
                    }
                }
            }
            printf("%d%c",v[i][j]," \n"[j==b]);
        }
    }
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 50. Testing Round #11 A. Up the hill 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/491/A

Testing Round #11 A. Up the hill 

Hiking club "Up the hill" just returned from a walk. Now they are trying 
to remember which hills they've just walked through.

It is known that there were N stops, all on different integer heights 
between 1 and N kilometers (inclusive) above the sea level. On the first 
day they've traveled from the first stop to the second stop, on the second 
day they've traveled from the second to the third and so on, and on the 
last day they've traveled from the stop N - 1 to the stop N and 
successfully finished their expedition.

They are trying to find out which heights were their stops located at. 
They have an entry in a travel journal specifying how many days did they 
travel up the hill, and how many days did they walk down the hill.

Help them by suggesting some possible stop heights satisfying numbers from 
the travel journal.
*/
void run(){
    int a,b;scanf("%d%d",&a,&b);
    int n=a+b+1;
    vector<int> v;
    for(int i=1;i<=n;i++)v.push_back(i);
    reverse(v.begin()+a,v.end());
    for(int i=0;i<n;i++)printf("%d%c",v[i]," \n"[i==n-1]);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 51. Codeforces Round #267 (Div. 2) B. Fedor and New Game 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/467/B

Codeforces Round #267 (Div. 2) B. Fedor and New Game 

After you had helped George and Alex to move in the dorm, they went to 
help their friend Fedor play a new computer game «Call of Soldiers 3».

The game has (m + 1) players and n types of soldiers in total. Players 
«Call of Soldiers 3» are numbered form 1 to (m + 1). Types of soldiers are 
numbered from 0 to n - 1. Each player has an army. Army of the i-th player 
can be described by non-negative integer x_i. Consider binary 
representation of x_i: if the j-th bit of number x_i equal to one, then 
the army of the i-th player has soldiers of the j-th type. 

Fedor is the (m + 1)-th player of the game. He assume that two players can 
become friends if their armies differ in at most k types of soldiers (in 
other words, binary representations of the corresponding numbers differ in 
at most k bits). Help Fedor and count how many players can become his 
friends.
*/
#define MAXN 1100
int x[MAXN];
void run(){
    int n,m,k;scanf("%d%d%d",&n,&m,&k);
    for(int i=1;i<=m+1;i++){
        scanf("%d",&x[i]);
    }
    int res=0;
    for(int i=1;i<=m;i++){
        if(__builtin_popcount(x[i]^x[m+1])<=k)res++;
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

# 52. Codeforces Round #644 (Div. 3) C. Similar Pairs 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1360/C

Codeforces Round #644 (Div. 3) C. Similar Pairs 

We call two numbers x and y similar if they have the same parity (the same 
remainder when divided by 2), or if |x-y|=1. For example, in each of the 
pairs (2, 6), (4, 3), (11, 7), the numbers are similar to each other, and 
in the pairs (1, 4), (3, 12), they are not.

You are given an array a of n (n is even) positive integers. Check if 
there is such a partition of the array into pairs that each element of the 
array belongs to exactly one pair and the numbers in each pair are similar 
to each other.

For example, for the array a = [11, 14, 16, 12], there is a partition into 
pairs (11, 12) and (14, 16). The numbers in the first pair are similar 
because they differ by one, and in the second pair because they are both 
even.
*/
#define MAXN 55
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    int odd=0,even=0;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(a[i]%2)odd++;
        else even++;
    }
    sort(a,a+n);
    int adj=0;
    for(int i=0;i<n-1;i++){
        if(abs(a[i]-a[i+1])==1){
            adj++;
            i++;
        }
    }
    odd-=adj;
    even-=adj;
    if((odd%2+even%2)>adj*2){
        printf("NO\n");
        return;
    }
    printf("YES\n");

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

# 53. Educational Codeforces Round 86 (Rated for Div. 2) B. Binary Period 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1342/B

Educational Codeforces Round 86 (Rated for Div. 2) B. Binary Period 

Let's say string s has period k if s_i = s_{i + k} for all i from 1 to |s| 
- k (|s| means length of string s) and k is the minimum positive integer 
with this property.

Some examples of a period: for s="0101" the period is k=2, for s="0000" 
the period is k=1, for s="010" the period is k=2, for s="0011" the period 
is k=4.

You are given string t consisting only of 0's and 1's and you need to find 
such string s that:

 String s consists only of 0's and 1's; 

 The length of s doesn't exceed 2 \cdot |t|; 

 String t is a subsequence of string s; 

 String s has smallest possible period among all strings that meet 
conditions 1—3. 

Let us recall that t is a subsequence of s if t can be derived from s by 
deleting zero or more elements (any) without changing the order of the 
remaining elements. For example, t="011" is a subsequence of s="10101".
*/
void run(){
    string s;cin>>s;
    int n=s.length();
    int z=0,o=0;
    for(int i=0;i<n;i++){
        if(s[i]=='0')z++;
        else o++;
    }
    if(!z||!o){
        cout<<s<<"\n";
        return;
    }
    string t;
    for(int i=0;i<n;i++){
        if(i==n-1||s[i]!=s[i+1])t+=s[i];
        else {
            t+=s[i];
            t+='0'+(s[i]=='0');
        }
    }
    cout<<t<<"\n";
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

# 54. Codeforces Round #651 (Div. 2) B. GCD Compression 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1370/B

Codeforces Round #651 (Div. 2) B. GCD Compression 

Ashish has an array a of consisting of 2n positive integers. He wants to 
compress a into an array b of size n-1. To do this, he first discards 
exactly 2 (any two) elements from a. He then performs the following 
operation until there are no elements left in a: 

 Remove any two elements from a and append their sum to b. 

The compressed array b has to have a special property. The greatest common 
divisor (\mathrm{gcd}) of all its elements should be greater than 1.

Recall that the \mathrm{gcd} of an array of positive integers is the 
biggest integer that is a divisor of all integers in the array.

It can be proven that it is always possible to compress array a into an 
array b of size n-1 such that gcd(b_1, b_2..., b_{n-1}) > 1. 

Help Ashish find a way to do so.
*/
#define MAXN 10001
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    vector<int> odd,even;
    for(int i=1;i<=2*n;i++){
        scanf("%d",&a[i]);
        if(a[i]%2)odd.push_back(i);
        else even.push_back(i);
    }
    int cnt=0;
    for(int i=0;i<odd.size();i+=2){
        if(cnt>=2*n-2)break;
        if(i+1<odd.size()){
            printf("%d %d\n",odd[i],odd[i+1]);
            cnt+=2;
        }
    }
    for(int i=0;i<even.size();i+=2){
        if(cnt>=2*n-2)break;
        if(i+1<even.size()){
            printf("%d %d\n",even[i],even[i+1]);
            cnt+=2;
        }
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

# 55. Codeforces Round #764 (Div. 3) C. Division by Two and Permutation 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1624/C

Codeforces Round #764 (Div. 3) C. Division by Two and Permutation 

You are given an array a consisting of n positive integers. You can 
perform operations on it.

In one operation you can replace any element of the array a_i with \lfloor 
\frac{a_i}{2} \rfloor, that is, by an integer part of dividing a_i by 2 
(rounding down).

See if you can apply the operation some number of times (possible 0) to 
make the array a become a permutation of numbers from 1 to n —that is, so 
that it contains all numbers from 1 to n, each exactly once.

For example, if a = [1, 8, 25, 2], n = 4, then the answer is yes. You 
could do the following:

 Replace 8 with \lfloor \frac{8}{2} \rfloor = 4, then a = [1, 4, 25, 2]. 

 Replace 25 with \lfloor \frac{25}{2} \rfloor = 12, then a = [1, 4, 12, 
2]. 

 Replace 12 with \lfloor \frac{12}{2} \rfloor = 6, then a = [1, 4, 6, 2]. 

 Replace 6 with \lfloor \frac{6}{2} \rfloor = 3, then a = [1, 4, 3, 2]. 
*/
#define MAXN 100
int a[MAXN];
void run(){
    int n;scanf("%d",&n);
    int res=0;
    unordered_map<int,int> mp;
    bool flag=false;
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        while(a[i]>n){
            a[i]/=2;
            res++;
        }
        while(a[i]!=0&&mp.find(a[i])!=mp.end()){
            a[i]/=2;
            res++;
        }
        if(a[i]==0)flag=true;
        else if(mp.find(a[i])==mp.end()){
            res++;
            mp[a[i]]=1;
        }
    }
    if(flag)printf("NO\n");
    else printf("YES\n");
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

# 56. Codeforces Round #665 (Div. 2) B. Ternary Sequence 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1401/B

Codeforces Round #665 (Div. 2) B. Ternary Sequence 

You are given two sequences a_1, a_2, \dots, a_n and b_1, b_2, \dots, b_n. 
Each element of both sequences is either 0, 1 or 2. The number of elements 
0, 1, 2 in the sequence a is x_1, y_1, z_1 respectively, and the number of 
elements 0, 1, 2 in the sequence b is x_2, y_2, z_2 respectively.

You can rearrange the elements in both sequences a and b however you like. 
After that, let's define a sequence c as follows:

c_i = \begin{cases} a_i b_i &amp; \mbox{if }a_i > b_i \\ 0 &amp; \mbox{if 
}a_i = b_i \\ -a_i b_i &amp; \mbox{if }a_i &lt; b_i \end{cases}

You'd like to make \sum_{i=1}^n c_i (the sum of all elements of the 
sequence c) as large as possible. What is the maximum possible sum?
*/
void run(){
    int x1,y1,z1,x2,y2,z2;
    scanf("%d%d%d%d%d%d",&x1,&y1,&z1,&x2,&y2,&z2);
    // x1 0 y1 1 z1 2
    // x2 0 y2 1 z2 2
    // 00000 11111 22222
    // 22111 11111 11100
    ll res=0;
    int p02=min(x1,z2);
    x1-=p02;
    z2-=p02;
    // 000 11111 22222
    // 111 11111 11100
    if(z2==0){
        printf("%d\n",min(z1,y2)*2);
        return;
    }
    // 111 11111 22222
    // 222 11111 11100

    // 11111 22222
    // 22222 11100
    int p21=min(z1,y2);
    z1-=p21;
    y2-=p21;
    res+=p21*2;
    // 111 11111
    // 222 11100

    int p22=min(z1,z2);
    z1-=p22;
    z2-=p22;
    // 111 22222
    // 222 22200
    res-=min(y1,z2)*2;
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

# 57. Educational Codeforces Round 107 (Rated for Div. 2) B. GCD Length 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1511/B

Educational Codeforces Round 107 (Rated for Div. 2) B. GCD Length 

You are given three integers a, b and c.

Find two positive integers x and y (x > 0, y > 0) such that: 

 the decimal representation of x without leading zeroes consists of a 
digits; 

 the decimal representation of y without leading zeroes consists of b 
digits; 

 the decimal representation of gcd(x, y) without leading zeroes consists 
of c digits. 

gcd(x, y) denotes the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor (GCD)

 of integers x and y.

Output x and y. If there are multiple answers, output any of them.
*/
void run(){
    int a,b,c;scanf("%d%d%d",&a,&b,&c);
    printf("1");
    for(int i=0;i<a-1;i++)printf("0");
    printf(" ");
    for(int i=0;i<b-c+1;i++)printf("1");
    for(int i=0;i<c-1;i++)printf("0");
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

# 58. Codeforces Round #669 (Div. 2) A. Ahahahahahahahaha 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1407/A

Codeforces Round #669 (Div. 2) A. Ahahahahahahahaha 

Alexandra has an even-length array a, consisting of 0s and 1s. The 
elements of the array are enumerated from 1 to n. She wants to remove at 
most \frac{n}{2} elements (where n — length of array) in the way that 
alternating sum of the array will be equal 0 (i.e. a_1 - a_2 + a_3 - a_4 + 
\dotsc = 0). In other words, Alexandra wants sum of all elements at the 
odd positions and sum of all elements at the even positions to become 
equal. The elements that you remove don't have to be consecutive.

For example, if she has a = [1, 0, 1, 0, 0, 0] and she removes 2nd and 4th 
elements, a will become equal [1, 1, 0, 0] and its alternating sum is 1 - 
1 + 0 - 0 = 0.

Help her!
*/
void run(){
    int n;scanf("%d",&n);
    int c0=0,c1=0;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(x)c1++;
        else c0++;
    }
    if(c1<=n/2){
        printf("%d\n",n/2);
        for(int i=0;i<n/2;i++)printf("0%c"," \n"[i==n/2-1]);
        return;
    }
    printf("%d\n",c1-c1%2);
    for(int i=0;i<c1-c1%2;i++)printf("1%c"," \n"[i==n/2-1]);
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

# 59. Codeforces Round #181 (Div. 2) A. Array 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/300/A

Codeforces Round #181 (Div. 2) A. Array 

Vitaly has an array of n distinct integers. Vitaly wants to divide this 
array into three non-empty sets so as the following conditions hold: 

 The product of all numbers in the first set is less than zero ( &lt; 0). 

 The product of all numbers in the second set is greater than zero ( > 0). 

 The product of all numbers in the third set is equal to zero. 

 Each number from the initial array must occur in exactly one set. 

Help Vitaly. Divide the given array.
*/
void run(){
    int n;scanf("%d",&n);
    vector<int> v1,v2,v3;
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(x<0)v1.push_back(x);
        if(x==0)v3.push_back(x);
        if(x>0)v2.push_back(x);
    }
    if(v2.empty()){
        v2.push_back(v1.back());v1.pop_back();
        v2.push_back(v1.back());v1.pop_back();
    }
    if(v1.size()%2==0){
        v3.push_back(v1.back());v1.pop_back();
    }
    printf("%d ",v1.size());for(const auto& x:v1)printf("%d ",x);printf("\n");
    printf("%d ",v2.size());for(const auto& x:v2)printf("%d ",x);printf("\n");
    printf("%d ",v3.size());for(const auto& x:v3)printf("%d ",x);printf("\n");
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 60. Educational Codeforces Round 99 (Rated for Div. 2) C. Ping-pong 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1455/C

Educational Codeforces Round 99 (Rated for Div. 2) C. Ping-pong 

Alice and Bob play ping-pong with simplified rules.

During the game, the player serving the ball commences a play. The server 
strikes the ball then the receiver makes a return by hitting the ball 
back. Thereafter, the server and receiver must alternately make a return 
until one of them doesn't make a return.

The one who doesn't make a return loses this play. The winner of the play 
commences the next play. Alice starts the first play.

Alice has x stamina and Bob has y. To hit the ball (while serving or 
returning) each player spends 1 stamina, so if they don't have any 
stamina, they can't return the ball (and lose the play) or can't serve the 
ball (in this case, the other player serves the ball instead). If both 
players run out of stamina, the game is over.

Sometimes, it's strategically optimal not to return the ball, lose the 
current play, but save the stamina. On the contrary, when the server 
commences a play, they have to hit the ball, if they have some stamina 
left.

Both Alice and Bob play optimally and want to, firstly, maximize their 
number of wins and, secondly, minimize the number of wins of their 
opponent.

Calculate the resulting number of Alice's and Bob's wins.
*/
void run(){
    // The best strategy for Bob is to not hit the ball for the
    // first x-1 turn.
    int x,y;scanf("%d%d",&x,&y);
    printf("%d %d\n",x-1,y);
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

# 61. Codeforces Round #674 (Div. 3) C. Increase and Copy 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1426/C

Codeforces Round #674 (Div. 3) C. Increase and Copy 

Initially, you have the array a consisting of one element 1 (a = [1]).

In one move, you can do one of the following things:

 Increase some (single) element of a by 1 (choose some i from 1 to the 
current length of a and increase a_i by one); 

 Append the copy of some (single) element of a to the end of the array 
(choose some i from 1 to the current length of a and append a_i to the end 
of the array). 

For example, consider the sequence of five moves:

 You take the first element a_1, append its copy to the end of the array 
and get a = [1, 1]. 

 You take the first element a_1, increase it by 1 and get a = [2, 1]. 

 You take the second element a_2, append its copy to the end of the array 
and get a = [2, 1, 1]. 

 You take the first element a_1, append its copy to the end of the array 
and get a = [2, 1, 1, 2]. 

 You take the fourth element a_4, increase it by 1 and get a = [2, 1, 1, 
3]. 

Your task is to find the minimum number of moves required to obtain the 
array with the sum at least n.

You have to answer t independent test cases.
*/
void run(){
    // I: increase
    // A: append
    // The pattern must be like: IIII..IIIAAAA.AAAA
    // The resule array must be like: [v,v,v,v,v...,v,v] (all the same)
    // Let k be the final length of the array
    // Then v = \lceil \frac{n}{k} \rceil
    // # of I = v - 1
    // # of A = k - 1
    // # of I+A = v + k - 2 = \lceil n/k \rceil + k - 2
    // Observe that the min is reached when k = \sqrt{n} for f(k) = n/k+k
    // Therefore we can try a few values around \sqrt{n}
    ll n;scanf("%lld",&n);
    auto f=[&](ll k){
        if(k<=0||k>n)return ll(2e9);
        return 1ll*((n+k-1)/k+k-2); // v + k - 2
    };
    ll k=sqrt(n);
    ll res=2e9;
    res=min(res,1ll*f(k));
    res=min(res,1ll*f(k-1));
    res=min(res,1ll*f(k+1));
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

# 62. Educational Codeforces Round 136 (Rated for Div. 2) B. Array Recovery 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1739/B

Educational Codeforces Round 136 (Rated for Div. 2) B. Array Recovery 

For an array of non-negative integers a of size n, we construct another 
array d as follows: d_1 = a_1, d_i = |a_i - a_{i - 1}| for 2 \le i \le n.

Your task is to restore the array a from a given array d, or to report 
that there are multiple possible arrays. 
*/
#define MAXN 110
int d[MAXN];
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%d",&d[i]);
    }
    int x=d[0];
    vector<int> res;
    res.push_back(x);
    for(int i=1;i<n;i++){
        if(d[i]==0){
            res.push_back(x);
            continue;
        }
        if(x-d[i]>=0){
            printf("-1\n");
            return;
        }
        x+=d[i];
        res.push_back(x);
    }
    for(const auto& x:res)printf("%d ",x);
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

# 63. Codeforces Global Round 9 A. Sign Flipping 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1375/A

Codeforces Global Round 9 A. Sign Flipping 

You are given n integers a_1, a_2, \dots, a_n, where n is odd. You are 
allowed to flip the sign of some (possibly all or none) of them. You wish 
to perform these flips in such a way that the following conditions hold:

 At least \frac{n - 1}{2} of the adjacent differences a_{i + 1} - a_i for 
i = 1, 2, \dots, n - 1 are greater than or equal to 0. 

 At least \frac{n - 1}{2} of the adjacent differences a_{i + 1} - a_i for 
i = 1, 2, \dots, n - 1 are less than or equal to 0. 

Find any valid way to flip the signs. It can be shown that under the given 
constraints, there always exists at least one choice of signs to flip that 
satisfies the required condition. If there are several solutions, you can 
find any of them.
*/
void run(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        int x;scanf("%d",&x);
        if(i%2&&x>0)x=-x;
        if(i%2==0&&x<0)x=-x;
        printf("%d ",x);
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

# 64. Codeforces Round 358 (Div. 2) A. Alyona and Numbers 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/682/A

Codeforces Round 358 (Div. 2) A. Alyona and Numbers 

After finishing eating her bun, Alyona came up with two integers n and m. 
She decided to write down two columns of integers — the first column 
containing integers from 1 to n and the second containing integers from 1 
to m. Now the girl wants to count how many pairs of integers she can 
choose, one from the first column and the other from the second column, 
such that their sum is divisible by 5.

Formally, Alyona wants to count the number of pairs of integers (x, y) 
such that 1 ≤ x ≤ n, 1 ≤ y ≤ m and <img align="middle" class="tex-formula" 
src="https://espresso.codeforces.com/172ce45a535a02e9d636dcf4c462a2a1df6c32
15.png" style="max-width: 100.0%;max-height: 100.0%;" /> equals 0.

As usual, Alyona has some troubles and asks you to help.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    int r=n%5;
    int k=n/5;
    ll res=0;
    for(int i=0;i<5;i++){
        if(i==0){
            res+=1ll*k*((i+m)/5);
        }else{
            res+=1ll*((i<=r?1:0)+k)*((i+m)/5);
        }
    }
    printf("%lld\n",res);
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```

# 65. Codeforces Round 808 (Div. 2) B. Difference of GCDs 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1708/B

Codeforces Round 808 (Div. 2) B. Difference of GCDs 

You are given three integers n, l, and r. You need to construct an array 
a_1,a_2,\dots,a_n (l\le a_i\le r) such that \gcd(i,a_i) are all distinct 
or report there's no solution.

Here \gcd(x, y) denotes the <a 
href="https://en.wikipedia.org/wiki/Greatest_common_divisor">greatest 
common divisor (GCD)

 of integers x and y.
*/
void run(){
    int n,l,r;scanf("%d%d%d",&n,&l,&r);
    vector<int> res;
    for(int i=1;i<=n;i++){
        int k=((l-1)/i+1)*i;
        if(k>r){
            puts("No\n");
            return;
        }
        res.push_back(k);
    }
    puts("Yes\n");
    for(auto x:res)printf("%d ",x);
    puts("");
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

# 66. Codeforces Round 620 (Div. 2) B. Longest Palindrome 

```cpp
#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
https://codeforces.com/problemset/problem/1304/B

Codeforces Round 620 (Div. 2) B. Longest Palindrome 

Returning back to problem solving, Gildong is now studying about 
palindromes. He learned that a palindrome is a string that is the same as 
its reverse. For example, strings "pop", "noon", "x", and "kkkkkk" are 
palindromes, while strings "moon", "tv", and "abab" are not. An empty 
string is also a palindrome.

Gildong loves this concept so much, so he wants to play with it. He has n 
distinct strings of equal length m. He wants to discard some of the 
strings (possibly none or all) and reorder the remaining strings so that 
the concatenation becomes a palindrome. He also wants the palindrome to be 
as long as possible. Please help him find one.
*/
void run(){
    int n,m;scanf("%d%d",&n,&m);
    string center;
    deque<string> dq;
    unordered_map<string,int> mp;
    for(int i=0;i<n;i++){
        string s;cin>>s;
        mp[s]++;
    }
    for(auto [k,v]:mp){
        int x=v;
        string t=k;
        reverse(t.begin(),t.end());
        if(t!=k){
            int y=mp[t];
            int d=min(x,y);
            mp[k]-=d;
            mp[t]-=d;
            for(int i=0;i<d;i++){
                dq.push_front(k);
                dq.push_back(t);
            }
        }
        if(k==t){
            if(mp[k]>1){
                int d=mp[k]/2;
                mp[k]-=d;
                mp[t]-=d;
                for(int i=0;i<d;i++){
                    dq.push_front(k);
                    dq.push_back(t);
                }
            }
            if(mp[k])center=k,mp[k]--;
            else if(mp[t])center=t,mp[t]--;
        }
    }
    int z=dq.size();
    cout<<z*m+center.size()<<"\n";
    for(int i=0;i<z/2;i++){
        cout<<dq.front();
        dq.pop_front();
    }
    cout<<center;
    for(int i=z/2;i<z;i++){
        cout<<dq.front();
        dq.pop_front();
    }
    cout<<"\n";
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    run();
}
```