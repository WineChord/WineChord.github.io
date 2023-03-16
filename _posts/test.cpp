// O(sqrt(n))
#include <iostream>
#include <algorithm>
using namespace std;
bool isp(int x)
{
    if (x < 2) return false;
    for (int i = 2; i <= x / i; i ++ )
        if (x % i == 0)
            return false;
    return true;
}

int main()
{
    int n;
    cin >> n;
    while (n -- )
    {
        int x;
        cin >> x;
        if (isp(x)) puts("Yes");
        else puts("No");
    }
    return 0;
}