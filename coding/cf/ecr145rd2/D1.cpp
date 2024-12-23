#include <bits/stdc++.h>
 
using i64 = long long;
 
constexpr i64 X = 1E12;
constexpr i64 inf = 1E18;
 
void update(i64 &a, i64 b) {
    if (a > b) {
        a = b;
    }
}
 
void solve() {
    std::string s;
    std::cin >> s;
    
    int n = s.size();
    
    std::vector dp(n + 1, std::array<i64, 2>{inf, inf});
    dp[0][0] = 0;
    for (int i = 0; i < n; i++) {
        for (int x = 0; x < 2; x++) {
            if (s[i] - '0' >= x) {
                update(dp[i + 1][s[i] - '0'], dp[i][x]);
            }
            update(dp[i + 1][x], dp[i][x] + X + 1);
            if (i + 1 < n && x <= s[i + 1] - '0' && s[i + 1] <= s[i]) {
                update(dp[i + 2][s[i] - '0'], dp[i][x] + X);
            }
        }
    }
    std::cout << std::min(dp[n][0], dp[n][1]) << "\n";
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