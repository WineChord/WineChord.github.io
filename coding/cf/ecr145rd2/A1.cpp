#include <bits/stdc++.h>
 
using i64 = long long;
 
/*
We can count the occurence of each color. There are only 5 ways to divide 4 bulbs into different colors: 1111, 112, 22, 13, 4. The answer for these situations are 4, 4, 4, 6 and -1.
*/
void solve() {
    std::string s;
    std::cin >> s;
    
    std::sort(s.begin(), s.end());
    
    if (s[0] == s[3]) {
        std::cout << -1 << "\n";
    } else if (s[0] == s[2] || s[1] == s[3]) {
        std::cout << 6 << "\n";
    } else {
        std::cout << 4 << "\n";
    }
}
 
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int t;
    std::cin >> t;
    
    while (t--) {
        solve();
    }
    
    return 0;
}
