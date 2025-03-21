#include <bits/stdc++.h>
 
using i64 = long long;
 
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int n, a, b;
    std::cin >> n >> a >> b;
    
    std::vector<int> v(n);
    for (int i = 0; i < n; i++) {
        std::cin >> v[i];
    }
    
    std::vector ans(a + 1, std::vector<int>(b + 1));
    for (int s = 0; s <= a + b; s++) {
        int X = std::max(0, s - b);
        int Y = std::min(a, s);
        int lo = X, hi = Y;
        for (int i = n - 1; i >= 0; i--) {
            lo += v[i];
            hi += v[i];
            lo = std::max(lo, X);
            hi = std::min(hi, Y);
        }
        int flo = lo, fhi = hi;
        for (int i = 0; i < n; i++) {
            flo = std::max(std::min(flo - v[i], Y), X);
            fhi = std::max(std::min(fhi - v[i], Y), X);
        }
        for (int c = X; c <= Y; c++) {
            int d = s - c;
            if (c <= lo) {
                ans[c][d] = flo;
            } else if (c >= hi) {
                ans[c][d] = fhi;
            } else {
                ans[c][d] = flo + c - lo;
            }
        }
    }
    
    for (int c = 0; c <= a; c++) {
        for (int d = 0; d <= b; d++) {
            std::cout << ans[c][d] << " \n"[d == b];
        }
    }
    
    return 0;
}
