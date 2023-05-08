---
classes: wide2
title: "Algorithm Template"
excerpt: "Collections of algorithm template."
categories: 
  - coding
tags: 
  - template
toc: true
toc_sticky: true
mathjax: true
---
## Basic Algorithms

### Quick Sort

#### Template

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN];
void qs(int *a,int l,int r){
    if(l>=r)return;
    int x=a[l+r>>1],i=l-1,j=r+1;
    while(i<j){
        do i++;while(a[i]<x);
        do j--;while(a[j]>x);
        if(i<j)swap(a[i],a[j]);
    }
    qs(a,l,j);qs(a,j+1,r);
}
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    qs(a,0,n-1);
    for(int i=0;i<n;i++)printf("%d ",a[i]);
}
```

#### Top-K: Quick Select

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN];
int qs(int l,int r,int k){
    if(l==r)return a[l];
    int x=a[(l+r)>>1],i=l-1,j=r+1;
    while(i<j){
        do i++;while(a[i]<x);
        do j--;while(a[j]>x);
        if(i<j)swap(a[i],a[j]);
    }
    int cnt=j-l+1;
    if(k<=cnt)return qs(l,j,k);
    return qs(j+1,r,k-cnt);
}
int main(){
    // k starts from 1.
    int n,k;scanf("%d%d",&n,&k);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    printf("%d\n",qs(0,n-1,k));
}
```

### Merge Sort

#### Merge Sort Template

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN],tmp[MAXN];
void ms(int l,int r){
    if(l>=r)return;
    int m=l+r>>1;
    ms(l,m);ms(m+1,r);
    int k=0,i=l,j=m+1;
    while(i<=m&&j<=r)
        if(a[i]<=a[j])tmp[k++]=a[i++];
        else tmp[k++]=a[j++];
    while(i<=m)tmp[k++]=a[i++];
    while(j<=r)tmp[k++]=a[j++];
    for(int i=l,j=0;i<=r;i++,j++)a[i]=tmp[j];
}
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    ms(0,n-1);
    for(int i=0;i<n;i++)printf("%d ",a[i]);
    return 0;
}
```

#### Count Inversions

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using ll=long long;
int a[MAXN],tmp[MAXN];
ll ms(int l,int r){
    if(l>=r)return 0;
    ll res=0;int m=l+r>>1;
    res+=ms(l,m)+ms(m+1,r);
    int i=l,j=m+1,k=0;
    while(i<=m&&j<=r)
        if(a[i]<=a[j])tmp[k++]=a[i++];
        else{
            tmp[k++]=a[j++];
            res+=m-i+1;
        }
    while(i<=m)tmp[k++]=a[i++];
    while(j<=r)tmp[k++]=a[j++];
    for(int i=l,j=0;i<=r;i++,j++)a[i]=tmp[j];
    return res;
}
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    printf("%lld\n",ms(0,n-1));
    return 0;
}
```

### Binary Search

#### Integer

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN];
int main(){
    int n,q;scanf("%d%d",&n,&q);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    while(q--){
        int k;scanf("%d",&k);
        int l=0,r=n-1;
        while(l<r){
            int m=l+r>>1; // Type 1.
            if(a[m]>=k)r=m;
            else l=m+1;
        }
        if(a[r]!=k){
            printf("-1 -1\n");
            continue;
        }
        printf("%d ",r);
        l=0,r=n-1;
        while(l<r){
            int m=l+r+1>>1; // Type 2.
            if(a[m]<=k)l=m;
            else r=m-1;
        }
        printf("%d\n",r);
    }
    return 0;
}
```

#### Floating Point

```c++
#include<iostream>
#include<cstdio>
using namespace std;
int main(){
    // Find x s.t. x*x*x = n.
    double n;scanf("%lf",&n);
    double l=-10000,r=10000;
    while(r-l>1e-8){
        double m=(l+r)/2;
        if(m*m*m>=n)r=m;
        else l=m;
    }
    printf("%.6lf\n",r);
    return 0;
}
```

### High Precision

#### Add

```c++
#include<iostream>
#include<cstdio>
#include<vector>
using namespace std;
vector<int> add(vector<int>& a,vector<int>& b) {
    vector<int> c;
    if(a.size()<b.size())return add(b,a);
    int up=0;
    for(int i=0;i<a.size();i++){
        up+=a[i];
        if(i<b.size())up+=b[i];
        c.push_back(up%10);
        up/=10;
    }
    if(up)c.push_back(1);
    return c;
}
int main(){
    string a,b;
    cin>>a>>b;
    vector<int> A,B;
    for(int i=a.length()-1;i>=0;i--)A.push_back(a[i]-'0');
    for(int i=b.length()-1;i>=0;i--)B.push_back(b[i]-'0');
    auto C=add(A,B);
    for(int i=C.size()-1;i>=0;i--)printf("%d",C[i]);
    return 0;
}
```

#### Sub

```c++
#include<iostream>
#include<cstdio>
#include<vector>
using namespace std;
bool cmp(vector<int>& a,vector<int>& b){
    if(a.size()!=b.size())return a.size()>b.size();
    for(int i=a.size()-1;i>=0;i--)
        if(a[i]!=b[i])return a[i]>b[i];
    return true;
}
vector<int> sub(vector<int>& a,vector<int>& b) {
    vector<int> c;
    int t=0;
    for(int i=0;i<a.size();i++){
        t=a[i]-t;
        if(i<b.size())t-=b[i];
        c.push_back((t+10)%10);
        if(t<0)t=1;
        else t=0;
    }
    while(c.size()>1&&c.back()==0)c.pop_back();
    return c;
}
int main(){
    string a,b;
    cin>>a>>b;
    vector<int> A,B;
    for(int i=a.length()-1;i>=0;i--)A.push_back(a[i]-'0');
    for(int i=b.length()-1;i>=0;i--)B.push_back(b[i]-'0');
    if(cmp(A,B)){
        auto C=sub(A,B);
        for(int i=C.size()-1;i>=0;i--)printf("%d",C[i]);
    }else{
        auto C=sub(B,A);printf("-");
        for(int i=C.size()-1;i>=0;i--)printf("%d",C[i]);        
    }
    return 0;
}
```

#### Mul

```c++
#include<iostream>
#include<cstdio>
#include<vector>
using namespace std;
vector<int> mul(vector<int>& A,int b){
    vector<int> C;
    int up=0;
    for(int i=0;i<A.size()||up;i++){
        if(i<A.size())up+=A[i]*b;
        C.push_back(up%10);
        up/=10;
    }
    while(C.size()>1&&C.back()==0)C.pop_back();
    return C;
}
int main(){
    string a;int b;
    cin>>a>>b;
    vector<int> A;
    for(int i=a.length()-1;i>=0;i--)A.push_back(a[i]-'0');
    auto C=mul(A,b);
    for(int i=C.size()-1;i>=0;i--)printf("%d",C[i]);
}
```

#### Div

```c++
#include<iostream>
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
vector<int> div(vector<int>& A,int b, int& r){
    vector<int> C;
    r=0;
    for(int i=A.size()-1;i>=0;i--){
        r=r*10+A[i];
        C.push_back(r/b);
        r%=b;
    }
    reverse(C.begin(),C.end());
    while(C.size()>1&&C.back()==0)C.pop_back();
    return C;
}
int main(){
    string a;int b;cin>>a>>b;
    vector<int> A;
    for(int i=a.size()-1;i>=0;i--)A.push_back(a[i]-'0');
    int r;
    auto C=div(A,b,r);
    for(int i=C.size()-1;i>=0;i--)printf("%d",C[i]);
    printf("\n%d",r);
}
```

### Prefix Sum and Difference

#### One Dimensional Prefix Sum

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN];
int main(){
    int n,m;
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++){
        scanf("%d",&a[i]);
        a[i]+=a[i-1];
    }
    for(int i=0;i<m;i++){
        int l,r;scanf("%d%d",&l,&r);
        printf("%d\n",a[r]-a[l-1]);
    }
    return 0;
}
```

#### Two Dimensional Prefix Sum

```c++
#include<iostream>
#include<cstdio>
#define MAXN 1010
int a[MAXN][MAXN];
using namespace std;
int main(){
    int n,m,q;scanf("%d%d%d",&n,&m,&q);
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            scanf("%d",&a[i][j]);
            a[i][j]+=a[i-1][j]+a[i][j-1]-a[i-1][j-1];
        }
    }
    while(q--){
        int x1,y1,x2,y2;scanf("%d%d%d%d",&x1,&y1,&x2,&y2);
        printf("%d\n",a[x2][y2]-a[x2][y1-1]-a[x1-1][y2]+a[x1-1][y1-1]);
    }
    return 0;
}
```

#### One Dimensional Difference

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    for(int i=n;i>1;i--)a[i]-=a[i-1];
    while(m--){
        int l,r,c;scanf("%d%d%d",&l,&r,&c);
        a[l]+=c;a[r+1]-=c;
    }
    for(int i=1;i<=n;i++){
        a[i]+=a[i-1];
        printf("%d ",a[i]);
    }
    return 0;
}
```

#### Two Dimensional Difference

```c++
#include<iostream>
#include<cstdio>
#define MAXN 1010
using namespace std;
int a[MAXN][MAXN];
int b[MAXN][MAXN];
void insert(int x1,int y1,int x2,int y2,int c){
    b[x1][y1]+=c;
    b[x1][y2+1]-=c;
    b[x2+1][y1]-=c;
    b[x2+1][y2+1]+=c;
}
int main(){
    int n,m,q;scanf("%d%d%d",&n,&m,&q);
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            scanf("%d",&a[i][j]);
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            insert(i,j,i,j,a[i][j]);
    while(q--){
        int x1,y1,x2,y2,c;
        scanf("%d%d%d%d%d",&x1,&y1,&x2,&y2,&c);
        insert(x1,y1,x2,y2,c);
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            b[i][j]+=b[i-1][j]+b[i][j-1]-b[i-1][j-1];
            printf("%d ",b[i][j]);
        }
        printf("\n");
    }
    return 0;
}
```

### Two Pointers

#### Longest Substring Without Repeating Characters

```c++
#include<iostream>
#include<cstdio>
#include<unordered_map>
#define MAXN 100010
using namespace std;
int a[MAXN];
int main(){
    int n;scanf("%d",&n);
    unordered_map<int,int> m;
    int res=0;
    for(int i=1;i<=n;i++)scanf("%d",&a[i]);
    for(int i=1,j=1;i<=n;i++){
        if(m.find(a[i])!=m.end())j=max(j,m[a[i]]+1);
        m[a[i]]=i;
        res=max(res,i-j+1);
    }
    printf("%d\n",res);
    return 0;
}
```

#### Target Sum

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN],b[MAXN];
int main(){
    int n,m,x;scanf("%d%d%d",&n,&m,&x);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    for(int j=0;j<m;j++)scanf("%d",&b[j]);
    int j=m-1;
    for(int i=0;i<n;i++){
        while(j>=0&&a[i]+b[j]>x)j--;
        if(a[i]+b[j]==x){
            printf("%d %d\n",i,j);
            break;
        }
    }
    return 0;
}
```

#### Is Subsequence

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN],b[MAXN];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    for(int j=0;j<m;j++)scanf("%d",&b[j]);
    int i=0,j=0;
    while(i<n&&j<m){
        if(a[i]==b[j])i++;
        j++;
    }
    if(i==n)printf("Yes\n");
    else printf("No\n");
    return 0;
}
```

### Bit Operation

```c++
#include<iostream>
#include<cstdio>
using namespace std;
int lowbit(int x) {return x&(-x);}
int main(){
    int n;scanf("%d",&n);
    while(n--){
        int x;scanf("%d",&x);
        int res=0;
        while(x){
            res++;
            x=x&(x-1);
        }
        printf("%d ",res);
    }
    return 0;
}
```

### Discretization

```c++
#include<iostream>
#include<cstdio>
#include<vector>
#include<algorithm>
#define MAXN 500010
using namespace std;
int a[MAXN];
int main(){
    int n,m;scanf("%d%d",&n,&m);
    vector<int> alls;
    vector<pair<int,int>> add;
    vector<pair<int,int>> query;
    for(int i=0;i<n;i++){
        int x,c;scanf("%d%d",&x,&c);
        alls.push_back(x);
        add.push_back({x,c});
    }
    for(int i=0;i<m;i++){
        int l,r;scanf("%d%d",&l,&r);
        alls.push_back(l);
        alls.push_back(r);
        query.push_back({l,r});
    }
    sort(alls.begin(),alls.end());
    alls.erase(unique(alls.begin(),alls.end()),alls.end());
    auto find=[&](int x){
        int l=0,r=alls.size()-1;
        while(l<r){
            int m=(l+r)/2;
            if(alls[m]>=x)r=m;
            else l=m+1;
        }
        return r+1;
    };
    for(auto [x,c]:add){
        int idx=find(x);
        a[idx]+=c;
    }
    for(int i=1;i<=alls.size();i++)a[i]+=a[i-1];
    for(auto [l,r]:query){
        int ll=find(l),rr=find(r);
        printf("%d\n",a[rr]-a[ll-1]);
    }
    return 0;
}
```

### Merge Intervals

```c++
#include<iostream>
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
int main(){
    int n;scanf("%d",&n);
    vector<pair<int,int>> segs;
    for(int i=0;i<n;i++){
        int l,r;scanf("%d%d",&l,&r);
        segs.push_back({l,r});
    }
    sort(segs.begin(),segs.end());
    int st=-2e9,ed=-2e9;
    vector<pair<int,int>> res;
    for(auto [l,r]:segs){
        if(ed<l){
            if(st!=-2e9)res.push_back({st,ed});
            st=l;ed=r;
        }else ed=max(ed,r);
    }
    if(st!=-2e9)res.push_back({st,ed});
    printf("%d\n",res.size());
    return 0;
}
```

## Basic Data Structures

### Singly Linked List

```c++
#include<iostream>
#include<cstdio>
#include<string>
#define MAXN 100100
using namespace std;
int e[MAXN],ne[MAXN],head,idx;
void init(){
    head=-1;idx=0;
}
void add_to_head(int x){
    e[idx]=x;
    ne[idx]=head;
    head=idx;
    idx++;
}
void add(int k,int x){
    e[idx]=x;
    ne[idx]=ne[k];
    ne[k]=idx;
    idx++;
}
void remove(int k){
    ne[k]=ne[ne[k]];
}
int main(){
    int m;scanf("%d",&m);
    init();
    while(m--){
        char c; scanf(" %c ",&c);
        if(c=='H'){
            int x;scanf("%d",&x);
            add_to_head(x);
        }else if(c=='D'){
            int k;scanf("%d",&k);
            if(k==0)head=ne[head];
            else remove(k-1);
        }else{
            int k,x;scanf("%d%d",&k,&x);
            add(k-1,x);
        }
    }
    int k=head;
    while(k!=-1){
        printf("%d ",e[k]);
        k=ne[k];
    }
    return 0;
}
```

### Doubly Linked List

```c++
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int e[MAXN],l[MAXN],r[MAXN],idx;
void init(){
    l[1]=0;r[0]=1;idx=2;
}
void add(int a,int x){
    e[idx]=x;
    l[idx]=a;r[idx]=r[a];
    l[r[a]]=idx;r[a]=idx++;
}
void remove(int a){
    r[l[a]]=r[a];
    l[r[a]]=l[a];
}
int main(){
    int m;scanf("%d",&m);
    init();
    while(m--){
        char c;scanf(" %c ",&c);
        if(c=='L'){
            int x;scanf("%d",&x);
            add(0,x);
        }else if(c=='R'){
            int x;scanf("%d",&x);
            add(l[1],x);
        }else if(c=='D'){
            int k;scanf("%d",&k);
            remove(k+1);
        }else if(c=='I'){
            scanf(" %c ",&c);
            if(c=='L'){
                int k,x;scanf("%d%d",&k,&x);
                add(l[k+1],x);
            }else if(c=='R'){
                int k,x;scanf("%d%d",&k,&x);
                add(k+1,x);
            }
        }
    }
    for(int i=r[0];i!=1;i=r[i])printf("%d ",e[i]);
    return 0;
}
```

### Stack

```cpp
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int stk[MAXN],tt;
int main(){
    int m;scanf("%d",&m);
    while(m--){
        string s;cin>>s;
        if(s=="push"){
            int x;scanf("%d",&x);
            stk[++tt]=x;
        }else if(s=="pop"){
            tt--;
        }else if(s=="query"){
            printf("%d\n",stk[tt]);
        }else if(s=="empty"){
            if(tt==0)printf("YES\n");
            else printf("NO\n");
        }
    }
    return 0;
}
```

#### Evaluate Equations

```c++
#include<iostream>
#include<cstdio>
#include<stack>
#include<unordered_map>
using namespace std;
stack<int> num;
stack<char> op;
void eval(){
    int b=num.top();num.pop();
    int a=num.top();num.pop();
    char c=op.top();op.pop();
    int x;
    if(c=='+')x=a+b;
    if(c=='-')x=a-b;
    if(c=='*')x=a*b;
    if(c=='/')x=a/b;
    num.push(x);
}
int main(){
    unordered_map<char,int> pr{
        {'+',1},{'-',1},{'*',2},{'/',2}
    };
    string s;cin>>s;
    for(int i=0;i<s.length();i++){
        char c=s[i];
        if(isdigit(c)){
            int x=0,j=i;
            while(j<s.length()&&isdigit(s[j]))
                x=x*10+s[j++]-'0';
            i=j-1;
            num.push(x);
        }else if(c=='(')op.push(c);
        else if(c==')'){
            while(op.top()!='(')eval();
            op.pop();
        }else{
            while(op.size()&&pr[op.top()]>=pr[c])eval();
            op.push(c);
        }
    }
    while(op.size())eval();
    cout<<num.top()<<endl;
    return 0;
}
```

### Queue

```c++
#include<iostream>
#include<cstdio>
#include<string>
#define MAXN 100010
using namespace std;
int q[MAXN],hh=0,tt=-1;
int main(){
    int m;scanf("%d",&m);
    while(m--){
        string s;cin>>s;
        if(s=="push"){
            int x;cin>>x;
            q[++tt]=x;
        }else if(s=="pop"){
            hh++;
        }else if(s=="empty"){
            if(hh==tt+1)printf("YES\n");
            else printf("NO\n");
        }else if(s=="query"){
            printf("%d\n",q[hh]);
        }
    }
    return 0;
}
```

### Monotonic Stack

```cpp
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int stk[MAXN],tt=0;
int main(){
    int n;cin>>n;
    for(int i=0;i<n;i++){
        int x;cin>>x;
        while(tt&&stk[tt]>=x)tt--;
        if(tt)printf("%d ",stk[tt]);
        else printf("-1 ");
        stk[++tt]=x;
    }
    return 0;
}
```

### Monotonic Queue

```cpp
#include<iostream>
#include<cstdio>
#define MAXN 1000010
using namespace std;
int q[MAXN],hh=0,tt=-1;
int a[MAXN];
int main(){
    int n,k;scanf("%d%d",&n,&k);
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        if(q[hh]<i-k+1)hh++;
        while(hh<=tt&&a[i]<=a[q[tt]])tt--;
        q[++tt]=i;
        if(i+1>=k)printf("%d ",a[q[hh]]);
    }
    printf("\n");
    hh=0;tt=-1;
    for(int i=0;i<n;i++){
        if(q[hh]<i-k+1)hh++;
        while(hh<=tt&&a[i]>=a[q[tt]])tt--;
        q[++tt]=i;
        if(i+1>=k)printf("%d ",a[q[hh]]);
    }
    return 0;
}
```

### KMP

```cpp
#include<iostream>
#include<cstdio>
#define MAXN 100010 
using namespace std;
int nxt[MAXN];
int main(){
    string p,s;
    int n,m;cin>>n>>p>>m>>s;
    int i,j;
    j=nxt[0]=-1;
    i=0;
    while(i<n){
        while(j!=-1&&p[i]!=p[j])j=nxt[j];
        // nxt[++i]=++j;
        if(p[++i]==p[++j])nxt[i]=nxt[j];
        else nxt[i]=j;
    }
    i=j=0;
    while(i<m){
        while(j!=-1&&s[i]!=p[j])j=nxt[j];
        i++;j++;
        if(j>=n){
            printf("%d ",i-j);
            j=nxt[j];
        }
    }
    return 0;
}
```

### Trie

```cpp
// Count occurrence of string.
#include<iostream>
#include<cstdio>
#define MAXN 100020
using namespace std;
int ch[MAXN][26],cnt[MAXN],idx;
void insert(string& s){
    int p=0;
    for(char cc:s){
        int c=cc-'a';
        if(!ch[p][c])ch[p][c]=++idx;
        p=ch[p][c];
    }
    cnt[p]++;
}
int query(string& s){
    int p=0;
    for(char cc:s){
        int c=cc-'a';
        if(!ch[p][c])return 0;
        p=ch[p][c];
    }
    return cnt[p];
}
int main(){
    int n;cin>>n;
    while(n--){
        string t,s;cin>>t>>s;
        if(t=="I"){
            insert(s);
        }else{
            cout<<query(s)<<endl;
        }
    }
    return 0;
}
```

```cpp
// Max XOR sum.
#include<iostream>
#include<cstdio>
#define MAXN 3100010
using namespace std;
int a[MAXN],ch[MAXN][2],idx;
void insert(int x){
    int p=0;
    for(int i=30;i>=0;i--){
        int c=(x>>i)&1;
        if(!ch[p][c])ch[p][c]=++idx;
        p=ch[p][c];
    }
}
int query(int x){
    int p=0,res=0;
    for(int i=30;i>=0;i--){
        int c=(x>>i)&1;
        if(ch[p][!c]){
            p=ch[p][!c];
            res=(res<<1)+1;
        }else{
            p=ch[p][c];
            res=(res<<1)+0;
        }
    }
    return res;
}
int main(){
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        scanf("%d",&a[i]);
        insert(a[i]);
    }
    int res=0;
    for(int i=0;i<n;i++){
        res=max(res,query(a[i]));
    }
    printf("%d",res);
    return 0;
}
```

### Union-Find

```cpp
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int p[MAXN];
int find(int x){
    if(p[x]!=x)return p[x]=find(p[x]);
    return p[x];
}
void uni(int a,int b){
    p[find(a)]=p[find(b)];
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)p[i]=i;
    while(m--){
        char c;int a,b;
        scanf(" %c %d %d ",&c,&a,&b);
        if(c=='M'){
            uni(a,b);
        }else{
            if(find(a)==find(b))printf("Yes\n");
            else printf("No\n");
        }
    }
    return 0;
}
```

```cpp
// Union-Find with size.
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int p[MAXN],sz[MAXN];
int find(int x){
    if(p[x]!=x)return p[x]=find(p[x]);
    return p[x];
}
void uni(int a,int b){
    if(find(a)==find(b))return;
    sz[find(b)]+=sz[find(a)];
    p[find(a)]=p[find(b)];
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)sz[i]=1,p[i]=i;
    while(m--){
        string c;int a,b;
        cin>>c;
        if(c=="C"){
            cin>>a>>b;
            uni(a,b);
        }else if(c=="Q1"){
            cin>>a>>b;
            if(find(a)==find(b))printf("Yes\n");
            else printf("No\n");
        }else{
            cin>>a;
            printf("%d\n",sz[find(a)]);
        }
    }
    return 0;
}
```

```cpp
// Union-Find with path distance.
#include<iostream>
#include<cstdio>
#define MAXN 50050
using namespace std;
int p[MAXN],d[MAXN];
int find(int x){
    if(p[x]!=x){
        int t=find(p[x]);
        d[x]+=d[p[x]];
        p[x]=t;
    }
    return p[x];
}
int main(){
    int n,k;scanf("%d%d",&n,&k);
    for(int i=1;i<=n;i++)p[i]=i;
    int res=0;
    while(k--){
        int dd,x,y;scanf("%d%d%d",&dd,&x,&y);
        if(x>n||y>n){
            res++;
            continue;
        }
        if(dd==1){
            int px=find(x),py=find(y);
            if(px==py){
                if((d[x]-d[y])%3)res++;
            }else{
                p[px]=py;
                d[px]=d[y]-d[x];
            }
        }else{
            if(x==y){
                res++;
                continue;
            }
            int px=find(x),py=find(y);
            if(px==py){
                if((d[x]-d[y]-1)%3!=0)res++;
            }else{
                p[py]=px;
                d[py]=d[x]-d[y]-1;
            }
        }
    }
    printf("%d\n",res);
    return 0;
}
```

### Heap

```cpp
#include<iostream>
#include<cstdio>
#define MAXN 100010
using namespace std;
int a[MAXN];
void heapify(int i,int n){
    int l=2*i+1,r=2*i+2;
    int m=i;
    if(l<n&&a[l]<a[m])m=l;
    if(r<n&&a[r]<a[m])m=r;
    if(m!=i){
        swap(a[m],a[i]);
        heapify(m,n);
    }
}
int main(){
    int n,m;
    scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)scanf("%d",&a[i]);
    for(int i=n/2;i>=0;i--)heapify(i,n);
    for(int i=n-1;i>=n-m;i--){
        printf("%d ",a[0]);
        swap(a[i],a[0]);
        heapify(0,i);
    }
    return 0;
}
```

```cpp
// Heap with index modification
#include<iostream>
#include<cstdio>
#define MAXN 100100
// #define WINE
using namespace std;
int a[MAXN],n,idx,hp[MAXN],ph[MAXN];
void hswap(int x,int y){
    swap(ph[hp[x]],ph[hp[y]]);
    swap(hp[x],hp[y]);
    swap(a[x],a[y]);
}
void up(int i){
    while(i&&a[i]<a[(i-1)/2]){
        hswap(i,(i-1)/2);
        i=(i-1)/2;
    }
}
void down(int i){
    int l=2*i+1,r=2*i+2;
    int m=i;
    if(l<n&&a[l]<a[m])m=l;
    if(r<n&&a[r]<a[m])m=r;
    if(m!=i){
        hswap(m,i);
        down(m);
    }
}
void print(){
#ifdef WINE
    printf("print: ");
    for(int i=0;i<n;i++)printf("%d ",a[i]);
    printf("\n");
#endif
}
int main(){
    int m;scanf("%d",&m);
    while(m--){
        string c;cin>>c;
        if(c=="I"){
            int x;cin>>x;
            hp[n]=++idx; // heap idx => global idx 
            ph[hp[n]]=n; // global idx => heap idx
            a[n++]=x;
            up(n-1);
            print();
        }else if(c=="PM"){
            printf("%d\n",a[0]);
            print();
        }else if(c=="DM"){
            hswap(0,n-1);
            n--;
            down(0);
            print();
        }else if(c=="D"){
            int k;cin>>k;
            int hidx=ph[k];
            hswap(ph[k],n-1);
            n--;
            up(hidx);
            down(hidx);
            print();
        }else if(c=="C"){
            int k,v;cin>>k>>v;
            a[ph[k]]=v;
            up(ph[k]);down(ph[k]);
            print();
        }
    }
    return 0;
}
```

### Hashmap

#### Constructing Hashmap

```cpp
// Open Hashing (Separate chaining)
#include<bits/stdc++.h>
#define N 100003
using namespace std;
int h[N],e[N],ne[N],idx;
void insert(int x){
    int k=(x%N+N)%N;
    e[idx]=x;ne[idx]=h[k],h[k]=idx++;
}
bool find(int x){
    int k=(x%N+N)%N;
    for(int i=h[k];i!=-1;i=ne[i]){
        if(e[i]==x)return true;
    }
    return false;
}
int main(){
    int n;scanf("%d",&n);
    memset(h,-1,sizeof(h));
    for(int i=0;i<n;i++){
        char op;int x;
        scanf(" %c %d",&op,&x);
        if(op=='I')insert(x);
        else{
            if(find(x))puts("Yes");
            else puts("No");
        }
    }
    return 0;
}
```

```cpp
// Closed Hashing (Open Addressing)
#include<bits/stdc++.h>
#define N 200003
#define INF 0x3f3f3f3f
using namespace std;
int h[N];
int find(int x){
    int k=(x%N+N)%N;
    while(h[k]!=INF&&h[k]!=x)k=(k+1)%N;
    return k;
}
int main(){
    int n;scanf("%d",&n);
    memset(h,INF,sizeof(h));
    for(int i=0;i<n;i++){
        char op;int x;
        scanf(" %c %d",&op,&x);
        int k=find(x);
        if(op=='I')h[k]=x;
        else{
            if(h[k]!=INF)puts("Yes");
            else puts("No");
        }
    }
    return 0;
}
```

#### String Hashing

```cpp
// Given a string, query whether two substrings are equal.
#include<bits/stdc++.h>
#define P 131 // OR 13331
#define MAXN 100010
using namespace std;
using ull=unsigned long long;
ull p[MAXN],h[MAXN];
char s[MAXN];
ull get(int l,int r){
    return h[r]-h[l-1]*p[r-l+1];
}
int main(){
    int n,m;scanf("%d%d %s",&n,&m,s+1);
    p[0]=1;
    for(int i=1;i<=n;i++){
        p[i]=p[i-1]*P;
        h[i]=h[i-1]*P+s[i];
    }
    while(m--){
        int l1,r1,l2,r2;
        scanf("%d%d%d%d",&l1,&r1,&l2,&r2);
        if(get(l1,r1)==get(l2,r2))puts("Yes");
        else puts("No");
    }
    return 0;
}
```

## Searching and Graph

### DFS

```cpp
// Print every permutations of 1-n.
#include<bits/stdc++.h>
#define MAXN 100
using namespace std;
int n;
bool used[MAXN];
int a[MAXN];
void dfs(int k){
    if(k==n){
        for(int i=0;i<n;i++)printf("%d%c",a[i]," \n"[i==n-1]);
        return;
    }
    for(int i=1;i<=n;i++)if(!used[i]){
        used[i]=true;
        a[k]=i;
        dfs(k+1);
        used[i]=false;
    }
}
int main(){
    scanf("%d",&n);
    dfs(0);
    return 0;
}
```

```cpp
// N-Queens
#include<bits/stdc++.h>
#define MAXN 11
using namespace std;
char g[MAXN][MAXN];
int n;
bool cc[MAXN],d1[MAXN],d2[MAXN];
void dfs(int r){
    if(r==n){
        for(int i=0;i<n;i++)puts(g[i]);
        puts("");
        return;
    }
    for(int c=0;c<n;c++){
        if(cc[c]||d1[r+c]||d2[n-r+c])continue;
        cc[c]=d1[r+c]=d2[n-r+c]=true;
        g[r][c]='Q';
        dfs(r+1);
        g[r][c]='.';
        cc[c]=d1[r+c]=d2[n-r+c]=false;
    }
}

int main(){
    scanf("%d",&n);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            g[i][j]='.';
    dfs(0);
    return 0;
}
```

```cpp
// N-Queues: another approach
#include<bits/stdc++.h>
#define MAXN 11
using namespace std;
char g[MAXN][MAXN];
int n;
bool rr[MAXN],cc[MAXN],d1[MAXN],d2[MAXN];
void dfs(int r,int c,int t){
    if(c==n)c=0,r++;
    if(r==n){
        if(t==n){
            for(int i=0;i<n;i++)puts(g[i]);
            puts("");
        }
        return;
    }
    dfs(r,c+1,t);
    if(rr[r]||cc[c]||d1[r+c]||d2[n-r+c])return;
    rr[r]=cc[c]=d1[r+c]=d2[n-r+c]=true;
    g[r][c]='Q';
    dfs(r,c+1,t+1);
    g[r][c]='.';
    rr[r]=cc[c]=d1[r+c]=d2[n-r+c]=false;
}

int main(){
    scanf("%d",&n);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            g[i][j]='.';
    dfs(0,0,0);
    return 0;
}
```

### BFS

```cpp
// Walk through a maze.
#include<bits/stdc++.h>
#define MAXN 110
#define INF 0x3f3f3f3f
using namespace std;
using pii=pair<int,int>;
int a[MAXN][MAXN];
int d[MAXN][MAXN];
int dx[4]={1,-1,0,0};
int dy[4]={0,0,1,-1};
int main(){
    int n,m;
    scanf("%d%d",&n,&m);
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            scanf("%d",&a[i][j]);
    memset(d,INF,sizeof(d));
    d[0][0]=0;
    queue<pii> q;
    q.push({0,0});
    while(!q.empty()){
        auto [x,y]=q.front();q.pop();
        for(int i=0;i<4;i++){
            int nx=x+dx[i];
            int ny=y+dy[i];
            if(nx<0||ny<0||nx>=n||ny>=m)continue;
            if(a[nx][ny]||d[nx][ny]!=INF)continue;
            d[nx][ny]=d[x][y]+1;
            q.push({nx,ny});
        }
    }
    printf("%d\n",d[n-1][m-1]);
}
```

```cpp
// Eight Digit
#include<bits/stdc++.h>
using namespace std;
int dx[4]={0,0,1,-1};
int dy[4]={1,-1,0,0};
int main(){
    string s;
    for(int i=0;i<9;i++){
        char c;scanf(" %c ",&c);
        s+=c;
    }
    queue<string> q;
    q.push(s);
    unordered_map<string,int> d;
    d[s]=0;
    string end="12345678x";
    while(q.size()){
        auto t=q.front();q.pop();
        int dis=d[t];
        // cout<<"# "<<t<<" "<<dis<<endl;
        if(t==end){
            printf("%d\n",dis);
            return 0;
        }
        int k=t.find('x');
        int x=k/3,y=k%3;
        for(int i=0;i<4;i++){
            int nx=x+dx[i];
            int ny=y+dy[i];
            if(nx<0||ny<0||nx>=3||ny>=3)continue;
            swap(t[k],t[nx*3+ny]);
            if(d.find(t)==d.end()){
                d[t]=dis+1;
                q.push(t);
            }
            swap(t[k],t[nx*3+ny]);
        }
    }
    printf("-1\n");
}
```


### DFS for Trees and Graphs

```cpp
// The center of gravity of a tree.
#include<bits/stdc++.h>
#define MAXN 200010
using namespace std;
int h[MAXN],e[MAXN],ne[MAXN],idx;
bool vis[MAXN];
void add(int u,int v){
    e[idx]=v;ne[idx]=h[u];h[u]=idx++;
}
int main(){
    int n;scanf("%d",&n);
    memset(h,-1,sizeof(h));
    for(int i=0;i<n-1;i++){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);add(v,u);
    }
    int res=n;
    function<int(int)> dfs=[&](int u){
        vis[u]=true;
        int sum=1,cnt=0;
        for(int i=h[u];i!=-1;i=ne[i]){
            int v=e[i];
            if(vis[v])continue;
            int s=dfs(v);
            cnt=max(cnt,s);
            sum+=s;
        }
        cnt=max(cnt,n-sum);
        res=min(res,cnt);
        return sum;
    };
    dfs(1);
    printf("%d\n",res);
    return 0;
}
```

### BFS for Trees and Graphs

```cpp
// Levels of points in a graph.
#include<bits/stdc++.h>
#define MAXN 200020
#define INF 0x3f3f3f3f
using namespace std;
int h[MAXN],e[MAXN],ne[MAXN],idx;
int d[MAXN];
void add(int u,int v){
    e[idx]=v;ne[idx]=h[u];h[u]=idx++;
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);
    }
    queue<int> q;
    q.push(1);
    memset(d,-1,sizeof(d));
    d[1]=0;
    while(!q.empty()){
        auto u=q.front();q.pop();
        for(int i=h[u];i!=-1;i=ne[i]){
            int v=e[i];
            if(d[v]!=-1)continue;
            d[v]=d[u]+1;
            q.push(v);
        }
    }
    printf("%d\n",d[n]);
}
```

### Topological Sort

```cpp
// Output a topological sort of a graph.
#include<bits/stdc++.h>
#define MAXN 200020
using namespace std;
int h[MAXN],e[MAXN],ne[MAXN],idx;
int in[MAXN];
bool vis[MAXN];
void add(int u,int v){
    e[idx]=v;ne[idx]=h[u];h[u]=idx++;
}
int main(){
    int n,m;scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);
        in[v]++;
    }
    queue<int> q;
    vector<int> res;
    for(int i=1;i<=n;i++)if(!in[i])q.push(i);
    while(!q.empty()){
        int u=q.front();q.pop();
        res.push_back(u);
        for(int i=h[u];i!=-1;i=ne[i]){
            int v=e[i];
            in[v]--;
            if(!in[v]){
                q.push(v);
            }
        }
    }
    if(res.size()==n){
        for(int i=0;i<n;i++)printf("%d%c",res[i]," \n"[i==n-1]);
        return 0;
    }
    printf("-1\n");
    return 0;
}
```

### Dijkstra

```
                        All weights are positive: Dijkstra O(n^2) or O(m\log(n))
                      /
            Single Source 
              /       \ 
Shortest Path          Have negative weights: Bellman-Ford O(nm) 
              \                            or SPFA general O(m) worst O(nm)
           Multiple Sources: Floyd O(n^3)
```

```cpp
// Dijkstra O(n^2)
#include<bits/stdc++.h>
#define MAXN 550
#define INF 0x3f3f3f3f
using namespace std;
int n,m;
bool vis[MAXN];
int g[MAXN][MAXN],d[MAXN];
int dijkstra(){
    memset(d,INF,sizeof(d));
    d[1]=0;
    for(int i=0;i<n-1;i++){
        int t=-1;
        for(int j=1;j<=n;j++)
            if(!vis[j]&&(t==-1||d[j]<d[t]))t=j;
        vis[t]=true;
        for(int j=1;j<=n;j++)d[j]=min(d[j],d[t]+g[t][j]);
    }
    if(d[n]==INF)return -1;
    return d[n];
}
int main(){
    scanf("%d%d",&n,&m);
    memset(g,INF,sizeof(g));
    while(m--){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        g[u][v]=min(g[u][v],w);
    }
    printf("%d\n",dijkstra());
}
```

```cpp
// Dijkstra O(m\log(n))
#include<bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MAXN 200020
using namespace std;
using pii=pair<int,int>;
int n,m,idx;
struct Edge{
    int v,w;
}e[MAXN*3];
bool vis[MAXN];
int h[MAXN],ne[MAXN],d[MAXN];
void add(int u,int v,int w){
    e[idx]={v,w};ne[idx]=h[u];h[u]=idx++;
}
int dijkstra(){
    memset(d,INF,sizeof(d));
    priority_queue<pii,vector<pii>,greater<pii>> q;
    q.push({0,1});
    d[1]=0;
    while(q.size()){
        auto [dis,u]=q.top();q.pop();
        if(vis[u])continue;vis[u]=true;
        for(int i=h[u];i!=-1;i=ne[i]){
            int v=e[i].v,w=e[i].w;
            if(d[v]>dis+w){
                d[v]=dis+w;
                q.push({d[v],v});
            }
        }
    }
    if(d[n]==INF)return -1;
    return d[n];
}
int main(){
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        add(u,v,w);
    }
    printf("%d\n",dijkstra());
}
```

### Bellman-Ford

```cpp
// Negative weights, at most k jumps to node n.
// Possible negative cycles.
// 1 <= n,k <= 500
// 1 <= m <= 10000
#include<bits/stdc++.h>
#define MAXN 550
#define MAXM 20020
#define INF 0x3f3f3f3f
using namespace std;
int n,m,k;
struct Edge{
    int u,v,w;
}e[MAXM];
int d[MAXN],bk[MAXN];
int bf(){
    memset(d,INF,sizeof(d));
    d[1]=0;
    for(int i=0;i<k;i++){
        memcpy(bk,d,sizeof(d));
        for(int j=0;j<m;j++){
            auto [u,v,w]=e[j];
            if(bk[u]!=INF)d[v]=min(d[v],bk[u]+w);
        }
    }
    return d[n];
}
int main(){
    scanf("%d%d%d",&n,&m,&k);
    for(int i=0;i<m;i++){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        e[i]={u,v,w};
    }
    int t=bf();
    if(t==INF)puts("impossible");
    else printf("%d\n",d[n]);
}
```

### SPFA

```cpp
// Negative weights.
// No negative cycles.
// 1 <= n,m <= 10^5
#include<bits/stdc++.h>
#define MAXN 200020
#define INF 0x3f3f3f3f
using namespace std;
struct Edge{
    int v,w;
}e[MAXN*3];
bool inq[MAXN];
int h[MAXN],ne[MAXN],idx,d[MAXN];
int n,m;
void add(int u,int v,int w){
    e[idx]={v,w};ne[idx]=h[u];h[u]=idx++;
}
int spfa(){
    memset(d,INF,sizeof(d));
    d[1]=0;
    queue<int> q;q.push(1);inq[1]=true;
    while(q.size()){
        auto u=q.front();q.pop();inq[u]=false;
        for(int i=h[u];i!=-1;i=ne[i]){
            auto [v,w]=e[i];
            if(d[v]>d[u]+w){
                d[v]=d[u]+w;
                if(!inq[v])q.push(v);
                inq[v]=true;
            }
        }
    }
    return d[n];
}
int main(){
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        add(u,v,w);
    }
    int t=spfa();
    if(t==INF)puts("impossible");
    else printf("%d\n",t);
}
```

```cpp
// Check negative cycle.
#include<bits/stdc++.h>
#define MAXN 200020
#define INF 0x3f3f3f3f
using namespace std;
struct Edge{
    int v,w;
}e[MAXN*3];
bool inq[MAXN];
int h[MAXN],ne[MAXN],idx,d[MAXN],cnt[MAXN];
int n,m;
void add(int u,int v,int w){
    e[idx]={v,w};ne[idx]=h[u];h[u]=idx++;
}
bool spfa(){
    queue<int> q;
    for(int i=1;i<=n;i++){
        q.push(i);inq[i]=true;
    }
    while(q.size()){
        auto u=q.front();q.pop();inq[u]=false;
        for(int i=h[u];i!=-1;i=ne[i]){
            auto [v,w]=e[i];
            if(d[v]>d[u]+w){
                d[v]=d[u]+w;
                cnt[v]=cnt[u]+1;
                if(cnt[v]>=n)return true;
                if(!inq[v])q.push(v);
                inq[v]=true;
            }
        }
    }
    return false;
}
int main(){
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        add(u,v,w);
    }
    if(spfa())puts("Yes");
    else puts("No");
}
```

### Floyd

```cpp
// Weights can be negative, multiple queries
// O(n^3)
#include<bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MAXN 220
using namespace std;
int g[MAXN][MAXN],n,m,k;
void floyd(){
    for(int k=1;k<=n;k++)
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++){
                if(g[i][k]==INF||g[k][j]==INF)continue;
                g[i][j]=min(g[i][j],g[i][k]+g[k][j]);
            }
}
int main(){
    scanf("%d%d%d",&n,&m,&k);
    for(int i=1;i<=n;i++)
        for(int j=1;j<=n;j++)
            if(i==j)g[i][j]=0;
            else g[i][j]=INF;
    while(m--){
        int u,v,w;
        scanf("%d%d%d",&u,&v,&w);
        g[u][v]=min(g[u][v],w);
    }
    floyd();
    while(k--){
        int u,v;scanf("%d%d",&u,&v);
        if(g[u][v]==INF)printf("impossible\n");
        else printf("%d\n",g[u][v]);
    }
}
```

### Prim

```cpp
// O(n^2)
#include<bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MAXN 550
using namespace std;
int g[MAXN][MAXN],d[MAXN],n,m;
bool vis[MAXN];
int prim(){
    memset(d,INF,sizeof(d));
    d[1]=0;
    int res=0;
    for(int i=0;i<n;i++){
        int k=-1;
        for(int j=1;j<=n;j++)
            if(!vis[j]&&(k==-1||d[k]>d[j]))k=j;
        if(d[k]==INF)return INF;
        res+=d[k];
        for(int j=1;j<=n;j++)d[j]=min(d[j],g[k][j]);
        vis[k]=true;
    }
    return res;
}
int main(){
    scanf("%d%d",&n,&m);
    memset(g,INF,sizeof(g));
    while(m--){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        g[u][v]=g[v][u]=min(g[u][v],w);
    }
    int t=prim();
    if(t==INF)printf("impossible\n");
    else printf("%d\n",t);
}
```

```cpp
// O(m\log(n))
#include<bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MAXN 550
using namespace std;
using pii=pair<int,int>;
int g[MAXN][MAXN],d[MAXN],n,m;
bool vis[MAXN];
int prim(){
    memset(d,INF,sizeof(d));
    priority_queue<pii,vector<pii>,greater<pii>> q;
    q.push({0,1});
    d[1]=0;
    int res=0,cnt=0;
    while(q.size()){
        auto [dis,u]=q.top();q.pop();
        if(vis[u])continue;vis[u]=true;
        cnt++;
        res+=dis;
        for(int v=1;v<=n;v++){
            if(g[u][v]==INF)continue;
            int w=g[u][v];
            if(d[v]>w){
                d[v]=w;
                q.push({d[v],v});
            }
        }
    }
    if(cnt!=n)return INF;
    return res;
}
int main(){
    scanf("%d%d",&n,&m);
    memset(g,INF,sizeof(g));
    while(m--){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        g[u][v]=g[v][u]=min(g[u][v],w);
    }
    int t=prim();
    if(t==INF)printf("impossible\n");
    else printf("%d\n",t);
}
```

### Kruskal

```cpp
// O(m\log(m))
// n <= 10^5
// m <= 2*10^5
#include<bits/stdc++.h>
#define MAXM 200200
#define MAXN 100100
#define INF 0x3f3f3f3f
using namespace std;
struct Edge{
    int u,v,w;
    // bool operator<(const Edge &W)const{
    //     return w<W.w;
    // }
}e[MAXM];
int n,m,p[MAXN];
int find(int x){
    if(p[x]==x)return x;
    return p[x]=find(p[x]);
}
int kruskal(){
    sort(e,e+m,[&](const Edge& a,const Edge& b){return a.w<b.w;});
    for(int i=1;i<=n;i++)p[i]=i;
    int res=0,cnt=0;
    for(int i=0;i<m;i++){
        auto [u,v,w]=e[i];
        u=find(u);v=find(v);
        if(u!=v){
            p[u]=v;
            res+=w;
            cnt++;
        }
    }
    if(cnt<n-1)return INF;
    return res;
}
int main(){
    scanf("%d%d",&n,&m);
    for(int i=0;i<m;i++){
        int u,v,w;scanf("%d%d%d",&u,&v,&w);
        e[i]={u,v,w};
    }
    int t=kruskal();
    if(t==INF)puts("impossible");
    else printf("%d\n",t);
}
```

### Coloring Method to Determine Bipartite Graph

```cpp
// O(n+m)
// Undirected graph.
// Has self-cycle and duplicat edges.
// n,m <= 10^5
//
// Bipartite Graph:
// * iif do not have cycles with odd number of nodes
#include<bits/stdc++.h>
#define MAXN 100010
#define MAXM 200020
using namespace std;
int n,m,h[MAXN],e[MAXM],ne[MAXM],idx,col[MAXN];
void add(int u,int v){
    e[idx]=v;ne[idx]=h[u];h[u]=idx++;
}
bool dfs(int u,int c){
    col[u]=c;
    for(int i=h[u];i!=-1;i=ne[i]){
        int v=e[i];
        if(!col[v]&&!dfs(v,3-c)||col[v]==c)return false;
    }
    return true;
}
int main(){
    scanf("%d%d",&n,&m);
    memset(h,-1,sizeof(h));
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);add(v,u);
    }
    bool flag=true;
    for(int i=1;i<=n;i++)
        if(!col[i]&&!dfs(i,1)){
            flag=false;
            break;
        }
    if(flag)puts("Yes");
    else puts("No");
}
```

### Hungarian Algorithm

```cpp
// Worst case O(nm)
#include<bits/stdc++.h>
#define MAXN 550
#define MAXM 100010
using namespace std;
int n1,n2,m,h[MAXN],e[MAXM],ne[MAXM],idx,match[MAXN],st[MAXN];
void add(int u,int v){
    e[++idx]=v;ne[idx]=h[u];h[u]=idx;
}
bool dfs(int u){
    for(int i=h[u];i;i=ne[i]){
        int v=e[i];
        if(st[v])continue;
        st[v]=1;
        if(!match[v]||dfs(match[v])){
            match[v]=u;
            return true;
        }
    }
    return false;
}
int main(){
    scanf("%d%d%d",&n1,&n2,&m);
    while(m--){
        int u,v;scanf("%d%d",&u,&v);
        add(u,v);
    }
    int res=0;
    for(int i=1;i<=n1;i++){
        memset(st,0,sizeof(st));
        if(dfs(i))res++;
    }
    printf("%d\n",res);
} 
```


## Basic Mathematics

### Prime

#### Primality Test

```cpp
// O(sqrt(n))
#include<bits/stdc++.h>
using namespace std;
bool isp(int x){
    if(x<2)return false;
    for(int i=2;i<=x/i;i++)if(x%i==0)return false;
    return true;
}
int main(){
    int n;cin>>n;
    while(n--){
        int x;cin>>x;
        if(isp(x))puts("Yes");
        else puts("No");
    }
}
```


#### Prime Factor Decomposition

```cpp
#include<bits/stdc++.h>
using namespace std;
void div(int x){
    for(int i=2;i<=x/i;i++){
        if(x%i==0){
            int s=0;
            while(x%i==0)x/=i,s++;
            cout<<i<<" "<<s<<"\n";
        }
    }
    if(x>1)cout<<x<<" 1\n";
    cout<<"\n";
}
int main(){
    int n;cin>>n;
    while(n--){
        int x;cin>>x;
        div(x);
    }
}
```

#### Sieves


```cpp
// Eratosthenes Sieve
// Time Complexity: $O(n\log\log(n))$.
#include<bits/stdc++.h>
#define MAXN 1000010
using namespace std;
int notp[MAXN],ps[MAXN],cnt;
int getp(int n){
    for(int i=2;i<=n;i++){
        if(notp[i])continue;
        ps[cnt++]=i;
        for(int j=i+i;j<=n;j+=i)notp[j]=1;
    }
}
int main(){
    int n;cin>>n;
    getp(n);
    cout<<cnt<<endl;
}
```


## String 

### Aho–Corasick algorithm

```cpp
// Count unique occurrence of patterns.
#include<bits/stdc++.h>
#define N 500050
using namespace std;
int ch[N][26],tot,cnt[N],fail[N];
void insert(const string& s){
    int u=0;
    for(auto cc:s){
        int c=cc-'a';
        if(!ch[u][c])ch[u][c]=++tot;
        u=ch[u][c];
    }
    cnt[u]++;
}
void build(){
    queue<int> q;
    for(int i=0;i<26;i++)if(ch[0][i])q.push(ch[0][i]);
    while(!q.empty()){
        auto u=q.front();q.pop();
        for(int i=0;i<26;i++){
            int v=ch[u][i];
            if(v)fail[v]=ch[fail[u]][i],q.push(v);
            else ch[u][i]=ch[fail[u]][i];
        }
    }
}
void run(){
    memset(ch,0,sizeof(ch));
    memset(fail,0,sizeof(fail));
    memset(cnt,0,sizeof(cnt));
    tot=0;
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        string s;cin>>s;insert(s);
    }
    build();
    string s;cin>>s;
    int res=0,u=0;
    for(auto cc:s){
        int c=cc-'a';
        u=ch[u][c];
        for(int i=u;i&&cnt[i]!=-1;i=fail[i]){
            res+=cnt[i];cnt[i]=-1;
        }
    }
    printf("%d\n",res);
}
int main(){
    run();
}
```

```cpp
// Count max occurrence of patterns.
#include<bits/stdc++.h>
#define N 206000
using namespace std;
int ch[N][26],tot,cnt[N],fail[N],idx,id[N],q[N],hh,tt,dup[N];
string ss[N];
int n;
void insert(const string& s){
    int u=0;
    for(auto cc:s){
        int c=cc-'a';
        if(!ch[u][c])ch[u][c]=++tot;
        u=ch[u][c];
    }
    id[idx++]=u;
}
void build(){
    for(int i=0;i<26;i++)if(ch[0][i])q[++tt]=ch[0][i];
    while(hh!=tt+1){
        auto u=q[hh++];
        for(int i=0;i<26;i++){
            int v=ch[u][i];
            if(v)fail[v]=ch[fail[u]][i],q[++tt]=v;
            else ch[u][i]=ch[fail[u]][i];
        }
    }
}
void run(){
    memset(ch,0,sizeof(ch));
    memset(fail,0,sizeof(fail));
    memset(cnt,0,sizeof(cnt));
    memset(id,0,sizeof(id));
    tot=idx=0;hh=0,tt=-1;
    for(int i=0;i<n;i++){
        string s;cin>>s;insert(s);
        ss[i]=s;
    }
    build();
    string s;cin>>s;
    int u=0;
    for(auto cc:s){
        int c=cc-'a';
        u=ch[u][c];
        cnt[u]++;
    }
    for(int i=tot-1;i>=0;i--)cnt[fail[q[i]]]+=cnt[q[i]];
    int mx=0;
    for(int i=0;i<idx;i++)mx=max(mx,cnt[id[i]]);
    printf("%d\n",mx);
    for(int i=0;i<idx;i++)if(mx==cnt[id[i]])cout<<ss[i]<<endl;
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    while(scanf("%d",&n)!=EOF&&n)run();
}
```

```cpp
// Count every occurrence of patterns.
#include<bits/stdc++.h>
#define N 1000060
using namespace std;
int ch[N][26],tot,cnt[N],fail[N],idx,id[N],q[N],hh,tt,dup[N];
void insert(const string& s){
    int u=0;
    for(auto cc:s){
        int c=cc-'a';
        if(!ch[u][c])ch[u][c]=++tot;
        u=ch[u][c];
    }
    id[idx++]=u;
}
void build(){
    for(int i=0;i<26;i++)if(ch[0][i])q[++tt]=ch[0][i];
    while(hh!=tt+1){
        auto u=q[hh++];
        for(int i=0;i<26;i++){
            int v=ch[u][i];
            if(v)fail[v]=ch[fail[u]][i],q[++tt]=v;
            else ch[u][i]=ch[fail[u]][i];
        }
    }
}
void run(){
    memset(ch,0,sizeof(ch));
    memset(fail,0,sizeof(fail));
    memset(cnt,0,sizeof(cnt));
    memset(id,0,sizeof(id));
    tot=idx=0;hh=0,tt=-1;
    int n;scanf("%d",&n);
    for(int i=0;i<n;i++){
        string s;cin>>s;insert(s);
    }
    build();
    string s;cin>>s;
    int u=0;
    for(auto cc:s){
        int c=cc-'a';
        int u=ch[u][c];
        cnt[u]++;
    }
    for(int i=tot-1;i>=0;i--)cnt[fail[q[i]]]+=cnt[q[i]];
    for(int i=0;i<idx;i++)printf("%d\n",cnt[id[i]]);
}
int main(){
    run();
}
```
