
In English По-русски


Educational Codeforces Round 145 (Rated for Div. 2)
A. Garland
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output
You have a garland consisting of 4
 colored light bulbs, the color of the 𝑖
-th light bulb is 𝑠𝑖
.

Initially, all the light bulbs are turned off. Your task is to turn all the light bulbs on. You can perform the following operation any number of times: select a light bulb and switch its state (turn it on if it was off, and turn it off if it was on). The only restriction on the above operation is that you can apply the operation to a light bulb only if the previous operation was applied to a light bulb of a different color (the first operation can be applied to any light bulb).

Calculate the minimum number of operations to turn all the light bulbs on, or report that this is impossible.

Input
The first line contains a single integer 𝑡
 (1≤𝑡≤104
) — the number of test cases.

The single line of each test case contains 𝑠
 — a sequence of 4
 characters, where each character is a decimal digit. The 𝑖
-th character denotes the color of the 𝑖
-th light bulb.

Output
For each test case, print one integer — the minimum number of operations to turn all the light bulbs on. If it is impossible to turn all the bulbs on, print -1.

Example
inputCopy
3
9546
0000
3313
outputCopy
4
-1
6
Note
In the first example, all the colors are different, so you can just turn all the bulbs on in 4
 operations.

In the second example, it is impossible to turn all the bulbs on, because after you switch one light bulb, it is impossible to turn the others on.

In the third example, you can proceed as follows: turn the first light bulb on, turn the third light bulb on, turn the fourth light bulb on, turn the third light bulb off, turn the second light bulb on, turn the third light bulb on.

B. Points on Plane
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output
You are given a two-dimensional plane, and you need to place 𝑛
 chips on it.

You can place a chip only at a point with integer coordinates. The cost of placing a chip at the point (𝑥,𝑦)
 is equal to |𝑥|+|𝑦|
 (where |𝑎|
 is the absolute value of 𝑎
).

The cost of placing 𝑛
 chips is equal to the maximum among the costs of each chip.

You need to place 𝑛
 chips on the plane in such a way that the Euclidean distance between each pair of chips is strictly greater than 1
, and the cost is the minimum possible.

Input
The first line contains one integer 𝑡
 (1≤𝑡≤104
) — the number of test cases. Next 𝑡
 cases follow.

The first and only line of each test case contains one integer 𝑛
 (1≤𝑛≤1018
) — the number of chips you need to place.

Output
For each test case, print a single integer — the minimum cost to place 𝑛
 chips if the distance between each pair of chips must be strictly greater than 1
.

Example
inputCopy
4
1
3
5
975461057789971042
outputCopy
0
1
2
987654321
Note
In the first test case, you can place the only chip at point (0,0)
 with total cost equal to 0+0=0
.

In the second test case, you can, for example, place chips at points (−1,0)
, (0,1)
 and (1,0)
 with costs |−1|+|0|=1
, |0|+|1|=1
 and |0|+|1|=1
. Distance between each pair of chips is greater than 1
 (for example, distance between (−1,0)
 and (0,1)
 is equal to 2‾√
). The total cost is equal to max(1,1,1)=1
.

In the third test case, you can, for example, place chips at points (−1,−1)
, (−1,1)
, (1,1)
, (0,0)
 and (0,2)
. The total cost is equal to max(2,2,2,0,2)=2
.

C. Sum on Subarrays
time limit per test2 seconds
memory limit per test512 megabytes
inputstandard input
outputstandard output
For an array 𝑎=[𝑎1,𝑎2,…,𝑎𝑛]
, let's denote its subarray 𝑎[𝑙,𝑟]
 as the array [𝑎𝑙,𝑎𝑙+1,…,𝑎𝑟]
.

For example, the array 𝑎=[1,−3,1]
 has 6
 non-empty subarrays:

𝑎[1,1]=[1]
;
𝑎[1,2]=[1,−3]
;
𝑎[1,3]=[1,−3,1]
;
𝑎[2,2]=[−3]
;
𝑎[2,3]=[−3,1]
;
𝑎[3,3]=[1]
.
You are given two integers 𝑛
 and 𝑘
. Construct an array 𝑎
 consisting of 𝑛
 integers such that:

all elements of 𝑎
 are from −1000
 to 1000
;
𝑎
 has exactly 𝑘
 subarrays with positive sums;
the rest (𝑛+1)⋅𝑛2−𝑘
 subarrays of 𝑎
 have negative sums.
Input
The first line contains one integer 𝑡
 (1≤𝑡≤5000
) — the number of test cases.

Each test case consists of one line containing two integers 𝑛
 and 𝑘
 (2≤𝑛≤30
; 0≤𝑘≤(𝑛+1)⋅𝑛2
).

Output
For each test case, print 𝑛
 integers — the elements of the array meeting the constraints. It can be shown that the answer always exists. If there are multiple answers, print any of them.

Example
inputCopy
4
3 2
2 0
2 2
4 6
outputCopy
1 -3 1
-13 -42
-13 42
-3 -4 10 -2
D. Binary String Sorting
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output
You are given a binary string 𝑠
 consisting of only characters 0 and/or 1.

You can perform several operations on this string (possibly zero). There are two types of operations:

choose two consecutive elements and swap them. In order to perform this operation, you pay 1012
 coins;
choose any element from the string and remove it. In order to perform this operation, you pay 1012+1
 coins.
Your task is to calculate the minimum number of coins required to sort the string 𝑠
 in non-decreasing order (i. e. transform 𝑠
 so that 𝑠1≤𝑠2≤⋯≤𝑠𝑚
, where 𝑚
 is the length of the string after applying all operations). An empty string is also considered sorted in non-decreasing order.

Input
The first line contains a single integer 𝑡
 (1≤𝑡≤104
) — the number of test cases.

The only line of each test case contains the string 𝑠
 (1≤|𝑠|≤3⋅105
), consisting of only characters 0 and/or 1.

The sum of lengths of all given strings doesn't exceed 3⋅105
.

Output
For each test case, print a single integer — the minimum number of coins required to sort the string 𝑠
 in non-decreasing order.

Example
inputCopy
6
100
0
0101
00101101
1001101
11111
outputCopy
1000000000001
0
1000000000000
2000000000001
2000000000002
0
Note
In the first example, you have to remove the 1
-st element, so the string becomes equal to 00.

In the second example, the string is already sorted.

In the third example, you have to swap the 2
-nd and the 3
-rd elements, so the string becomes equal to 0011.

In the fourth example, you have to swap the 3
-rd and the 4
-th elements, so the string becomes equal to 00011101, and then remove the 7
-th element, so the string becomes equal to 0001111.

In the fifth example, you have to remove the 1
-st element, so the string becomes equal to 001101, and then remove the 5
-th element, so the string becomes equal to 00111.

In the sixth example, the string is already sorted.

E. Two Tanks
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output
There are two water tanks, the first one fits 𝑎
 liters of water, the second one fits 𝑏
 liters of water. The first tank has 𝑐
 (0≤𝑐≤𝑎
) liters of water initially, the second tank has 𝑑
 (0≤𝑑≤𝑏
) liters of water initially.

You want to perform 𝑛
 operations on them. The 𝑖
-th operation is specified by a single non-zero integer 𝑣𝑖
. If 𝑣𝑖>0
, then you try to pour 𝑣𝑖
 liters of water from the first tank into the second one. If 𝑣𝑖<0
, you try to pour −𝑣𝑖
 liters of water from the second tank to the first one.

When you try to pour 𝑥
 liters of water from the tank that has 𝑦
 liters currently available to the tank that can fit 𝑧
 more liters of water, the operation only moves min(𝑥,𝑦,𝑧)
 liters of water.

For all pairs of the initial volumes of water (𝑐,𝑑)
 such that 0≤𝑐≤𝑎
 and 0≤𝑑≤𝑏
, calculate the volume of water in the first tank after all operations are performed.

Input
The first line contains three integers 𝑛,𝑎
 and 𝑏
 (1≤𝑛≤104
; 1≤𝑎,𝑏≤1000
) — the number of operations and the capacities of the tanks, respectively.

The second line contains 𝑛
 integers 𝑣1,𝑣2,…,𝑣𝑛
 (−1000≤𝑣𝑖≤1000
; 𝑣𝑖≠0
) — the volume of water you try to pour in each operation.

Output
For all pairs of the initial volumes of water (𝑐,𝑑)
 such that 0≤𝑐≤𝑎
 and 0≤𝑑≤𝑏
, calculate the volume of water in the first tank after all operations are performed.

Print 𝑎+1
 lines, each line should contain 𝑏+1
 integers. The 𝑗
-th value in the 𝑖
-th line should be equal to the answer for 𝑐=𝑖−1
 and 𝑑=𝑗−1
.

Examples
inputCopy
3 4 4
-2 1 2
outputCopy
0 0 0 0 0 
0 0 0 0 1 
0 0 1 1 2 
0 1 1 2 3 
1 1 2 3 4 
inputCopy
3 9 5
1 -2 2
outputCopy
0 0 0 0 0 0 
0 0 0 0 0 1 
0 1 1 1 1 2 
1 2 2 2 2 3 
2 3 3 3 3 4 
3 4 4 4 4 5 
4 5 5 5 5 6 
5 6 6 6 6 7 
6 7 7 7 7 8 
7 7 7 7 8 9 
Note
Consider 𝑐=3
 and 𝑑=2
 from the first example:

The first operation tries to move 2
 liters of water from the second tank to the first one, the second tank has 2
 liters available, the first tank can fit 1
 more liter. Thus, min(2,2,1)=1
 liter is moved, the first tank now contains 4
 liters, the second tank now contains 1
 liter.
The second operation tries to move 1
 liter of water from the first tank to the second one. min(1,4,3)=1
 liter is moved, the first tank now contains 3
 liters, the second tank now contains 2
 liter.
The third operation tries to move 2
 liter of water from the first tank to the second one. min(2,3,2)=2
 liters are moved, the first tank now contains 1
 liter, the second tank now contains 4
 liters.
There's 1
 liter of water in the first tank at the end. Thus, the third value in the fourth row is 1
.

F. Traveling in Berland
time limit per test2 seconds
memory limit per test256 megabytes
inputstandard input
outputstandard output
There are 𝑛
 cities in Berland, arranged in a circle and numbered from 1
 to 𝑛
 in clockwise order.

You want to travel all over Berland, starting in some city, visiting all the other cities and returning to the starting city. Unfortunately, you can only drive along the Berland Ring Highway, which connects all 𝑛
 cities. The road was designed by a very titled and respectable minister, so it is one-directional — it can only be traversed clockwise, only from the city 𝑖
 to the city (𝑖mod𝑛)+1
 (i.e. from 1
 to 2
, from 2
 in 3
, ..., from 𝑛
 to 1
).

The fuel tank of your car holds up to 𝑘
 liters of fuel. To drive from the 𝑖
-th city to the next one, 𝑎𝑖
 liters of fuel are needed (and are consumed in the process).

Every city has a fuel station; a liter of fuel in the 𝑖
-th city costs 𝑏𝑖
 burles. Refueling between cities is not allowed; if fuel has run out between cities, then your journey is considered incomplete.

For each city, calculate the minimum cost of the journey if you start and finish it in that city.

Input
The first line contains a single integer 𝑡
 (1≤𝑡≤104
) — the number of test cases.

The first line of each test case contains two integers 𝑛
 and 𝑘
 (3≤𝑛≤2⋅105
; 1≤𝑘≤109
) — the number of cities and the volume of fuel tank, respectively.

The second line contains 𝑛
 integers 𝑎1,𝑎2,…,𝑎𝑛
 (1≤𝑎𝑖≤𝑘
).

The third line contains 𝑛
 integers 𝑏1,𝑏2,…,𝑏𝑛
 (1≤𝑏𝑖≤2
).

The sum of 𝑛
 over all test cases doesn't exceed 2⋅105
.

Output
For each test case, print 𝑛
 integers, where the 𝑖
-th of them is equal to the minimum cost of the journey if you start and finish in the 𝑖
-th city.

Example
inputCopy
4
3 5
3 4 4
1 2 2
5 7
1 3 2 5 1
2 1 1 1 2
4 3
1 2 1 3
2 2 2 2
3 2
2 2 2
1 2 1
outputCopy
17 19 17 
13 12 12 12 14 
14 14 14 14 
8 8 8 
G. Prediction
time limit per test4 seconds
memory limit per test512 megabytes
inputstandard input
outputstandard output
Consider a tournament with 𝑛
 participants. The rating of the 𝑖
-th participant is 𝑎𝑖
.

The tournament will be organized as follows. First of all, organizers will assign each participant an index from 1
 to 𝑛
. All indices will be unique. Let 𝑝𝑖
 be the participant who gets the index 𝑖
.

Then, 𝑛−1
 games will be held. In the first game, participants 𝑝1
 and 𝑝2
 will play. In the second game, the winner of the first game will play against 𝑝3
. In the third game, the winner of the second game will play against 𝑝4
, and so on — in the last game, the winner of the (𝑛−2)
-th game will play against 𝑝𝑛
.

Monocarp wants to predict the results of all 𝑛−1
 games (of course, he will do the prediction only after the indices of the participants are assigned). He knows for sure that, when two participants with ratings 𝑥
 and 𝑦
 play, and |𝑥−𝑦|>𝑘
, the participant with the higher rating wins. But if |𝑥−𝑦|≤𝑘
, any of the two participants may win.

Among all 𝑛!
 ways to assign the indices to participants, calculate the number of ways to do this so that Monocarp can predict the results of all 𝑛−1
 games. Since the answer can be large, print it modulo 998244353
.

Input
The first line contains two integers 𝑛
 and 𝑘
 (2≤𝑛≤106
; 0≤𝑘≤109
).

The second line contains 𝑛
 integers 𝑎1,𝑎2,…,𝑎𝑛
 (0≤𝑎1≤𝑎2≤⋯≤𝑎𝑛≤109
).

Output
Print one integer — the number of ways to assign the indices to the participants so that Monocarp can predict the results of all 𝑛−1
 games.

Examples
inputCopy
4 3
7 12 17 21
outputCopy
24
inputCopy
3 7
4 9 28
outputCopy
4
inputCopy
4 1
1 2 3 4
outputCopy
0
inputCopy
4 1
1 2 2 4
outputCopy
12
inputCopy
16 30
8 12 15 27 39 44 49 50 51 53 58 58 59 67 68 100
outputCopy
527461297
Note
In the first example, a match with any pair of players can be predicted by Monocarp, so all 24
 ways to assign indices should be counted.

In the second example, suitable ways are [1,3,2]
, [2,3,1]
, [3,1,2
] and [3,2,1]
.

Codeforces (c) Copyright 2010-2023 Mike Mirzayanov
The only programming contests Web 2.0 platform