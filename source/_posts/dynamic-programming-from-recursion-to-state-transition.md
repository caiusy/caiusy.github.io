---
title: 动态规划完全图解：从重复递归到状态转移
date: 2026-07-13 23:50:00
updated: 2026-07-13 23:50:00
mathjax: true
description: "从 LC70 爬楼梯出发，用递归树、状态表和完整代码理解动态规划五步法，再迁移到线性 DP、网格 DP、0/1 与完全背包、子序列和股票状态机。"
categories:
  - 算法与数据结构
  - LeetCode
tags:
  - 动态规划
  - DP
  - LeetCode
  - Python
  - C++
type: deep-dive
difficulty: intermediate
review_status: published
---

> 动态规划不等于背公式。本文从一个会重复计算的递归开始，逐步得到状态定义、转移方程、初始化、遍历顺序和空间优化，并用 `n=5 -> 8` 的完整例子把每一步连起来。

<!-- more -->

## 1. 先用一句人话理解动态规划

动态规划（Dynamic Programming，DP）可以理解为：

> **把递归中反复出现的问题贴上标签，把答案写进表格；以后遇到相同标签，直接查表，不再重算。**

比如计算爬楼梯的方法数。暴力递归每次都会重新追问“到第 3 阶有多少种方法”；DP 则在第 3 阶贴一张纸写下答案，以后直接读取。

这条推理链比任何固定模板都重要：

```text
暴力递归
-> 发现重复子问题
-> 用状态表示子问题
-> 写出状态之间的关系
-> 按依赖顺序计算
-> 必要时压缩空间
```

![递归树压缩为动态规划状态链](/images/dynamic-programming-mastery/01_recursion_to_dp.png)

图中左侧的红色节点表示重复计算。右侧没有改变数学递推关系，只是把每种状态保留一次。因此复杂度能从指数级递归树降到线性状态数量。

## 2. 什么问题值得考虑 DP

DP 常见于三类目标：

| 目标 | 题目措辞 | 状态通常保存什么 |
|---|---|---|
| 最优值 | 最大、最小、最多、最少 | 当前范围内的最优答案 |
| 方案数 | 多少种方法、多少条路径 | 到达当前状态的方案数量 |
| 可行性 | 能否、是否存在 | `True / False` |

它通常还具备两个性质：

1. **重叠子问题**：同一个小问题会沿不同路径反复出现。
2. **最优子结构**：大问题可以由小问题答案正确组合得到。

### DP、贪心和回溯怎么区分

| 方法 | 保存什么 | 如何面对选择 | 典型任务 |
|---|---|---|---|
| DP | 每个状态的答案 | 比较所有合法来路，并合并相同状态 | 最优值、方案数、可行性 |
| 贪心 | 当前局部最优 | 做出选择后通常不回头 | 可证明局部最优导向全局最优 |
| 回溯 | 当前路径 | 选择、递归、撤销，枚举路径 | 输出具体方案或搜索小规模空间 |

记忆句：**要路径细节多回溯，要状态答案想 DP；只留眼前仍正确，才用贪心。**

## 3. DP 五问：写代码前先把中文说清楚

遇到一道 DP 题，先回答下面五问：

1. **状态**：`dp[i]` 或 `dp[i][j]` 的完整中文含义是什么？
2. **转移**：当前状态从哪些旧状态而来？
3. **初始化**：最小子问题是什么，答案是多少？
4. **顺序**：怎样计算才能保证依赖状态已经得到？
5. **答案**：返回末格、整张表的最大值，还是某个特定状态？

总口诀：

> **状态先说清，转移找来路；边界定起点，依赖定顺序；答案看题意，最后再压缩。**

### 为什么状态必须写成完整句子

“`dp[i]` 是最大值”几乎没有信息。至少要说清：

- 是前 `i` 个元素的最大值，还是必须以第 `i` 个元素结尾？
- 是否必须选择当前元素？
- `i` 表示元素下标，还是已处理的元素数量？

例如 LC53 最大子数组和必须定义为：`dp[i]` 是**以 `nums[i]` 结尾**的最大连续子数组和。正是“必须结尾”这个约束，让它能从 `dp[i-1]` 转移。

## 4. 同一个递推的四种实现

先看斐波那契式关系：

{% raw %}
$$
f(n)=f(n-1)+f(n-2)
$$
{% endraw %}

### 4.1 暴力递归

```python
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

代码贴近数学定义，但同一参数会重复计算，时间复杂度约为 `O(2^n)`。

### 4.2 记忆化递归：自顶向下

```python
def fib(n: int) -> int:
    memo = {}

    def dfs(x: int) -> int:
        if x <= 1:
            return x
        if x not in memo:
            memo[x] = dfs(x - 1) + dfs(x - 2)
        return memo[x]

    return dfs(n)
```

每个 `x` 只计算一次，时间和空间均为 `O(n)`。

### 4.3 递推填表：自底向上

```python
def fib(n: int) -> int:
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

记忆化是“从目标向下问”，递推是“从最小答案向上填”，二者使用相同的状态关系。

### 4.4 滚动变量

```python
def fib(n: int) -> int:
    if n <= 1:
        return n
    previous, current = 0, 1
    for _ in range(2, n + 1):
        previous, current = current, previous + current
    return current
```

只有确认当前状态只依赖前两项，才可以丢掉更早状态，把空间降为 `O(1)`。学习新模型时，建议先写完整表，再做压缩。

## 5. 端到端例子：LC70 爬楼梯

### 5.1 不从公式出发，从最后一步出发

每次走 1 阶或 2 阶。要到第 `i` 阶，最后一步只有两种互斥情况：

1. 从第 `i-1` 阶走 1 步；
2. 从第 `i-2` 阶走 2 步。

定义 `dp[i]` 为到达第 `i` 阶的方法数，得到：

{% raw %}
$$
dp[i]=dp[i-1]+dp[i-2]
$$
{% endraw %}

这不是凭空出现的公式，而是把所有可能的“最后一步”分类后相加。

### 5.2 用五问完成建模

| 问题 | LC70 的答案 |
|---|---|
| 状态 | `dp[i]` 表示到达第 `i` 阶的方法数 |
| 转移 | `dp[i] = dp[i-1] + dp[i-2]` |
| 初始化 | `dp[0] = 1`，`dp[1] = 1` |
| 顺序 | `i` 从 2 到 `n` |
| 返回 | `dp[n]` |

`dp[0] = 1` 表示一条“什么都不做的空方案”。它让“从第 0 阶一次走 2 步到第 2 阶”能被正确计数。

### 5.3 手推 `n=5 -> 8`

![LC70 爬楼梯状态表](/images/dynamic-programming-mastery/02_climbing_stairs_table.png)

| `i` | 两条来路 | 计算 | `dp[i]` |
|---|---|---|---:|
| 0 | 空方案 | 初始化 | 1 |
| 1 | 从 0 走 1 步 | 初始化 | 1 |
| 2 | 第 1 阶、第 0 阶 | `1 + 1` | 2 |
| 3 | 第 2 阶、第 1 阶 | `2 + 1` | 3 |
| 4 | 第 3 阶、第 2 阶 | `3 + 2` | 5 |
| 5 | 第 4 阶、第 3 阶 | `5 + 3` | 8 |

### 5.4 Python 完整表版本

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]
```

### 5.5 Python 空间优化版本

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        previous, current = 1, 1

        for _ in range(2, n + 1):
            previous, current = current, previous + current

        return current
```

### 5.6 C++ 空间优化版本

```cpp
class Solution {
public:
    int climbStairs(int n) {
        int previous = 1;
        int current = 1;

        for (int step = 2; step <= n; ++step) {
            int next = previous + current;
            previous = current;
            current = next;
        }

        return current;
    }
};
```

时间复杂度 `O(n)`；完整表空间 `O(n)`，滚动变量空间 `O(1)`。

记忆口诀：**最后一步一或二，两条来路加一起；起点一、首阶一，前两项向后移。**

## 6. 线性 DP：从相加到“选或不选”

LC198 打家劫舍中，相邻房屋不能同时偷。定义 `dp[i]` 为考虑前 `i` 间房时能偷到的最大金额。

面对当前房屋：

```text
不偷：dp[i - 1]
偷：  dp[i - 2] + nums[i - 1]
```

所以：

{% raw %}
$$
dp[i]=\max(dp[i-1],\ dp[i-2]+nums[i-1])
$$
{% endraw %}

```python
def rob(nums: list[int]) -> int:
    two_back = 0
    one_back = 0
    for money in nums:
        two_back, one_back = one_back, max(one_back, two_back + money)
    return one_back
```

对于 `[2,7,9,3,1]`，状态依次得到 `2, 7, 11, 11, 12`。

口诀：**当前房，偷不偷；偷看隔壁的隔壁，不偷沿用前一屋。**

## 7. 网格 DP：依赖箭头决定填表方向

LC62 不同路径中，机器人只能向右或向下。当前格最后一步只能来自上方或左方：

{% raw %}
$$
dp[i][j]=dp[i-1][j]+dp[i][j-1]
$$
{% endraw %}

`3 x 3` 网格的状态表是：

```text
1  1  1
1  2  3
1  3  6
```

一维压缩：

```python
def uniquePaths(m: int, n: int) -> int:
    dp = [1] * n
    for _ in range(1, m):
        for column in range(1, n):
            dp[column] += dp[column - 1]
    return dp[-1]
```

更新前的 `dp[column]` 表示上方格，更新后的 `dp[column-1]` 表示左方格。

口诀：**只能右下走，当前上加左；一行一列一条路，答案落在右下角。**

## 8. 背包 DP：方向决定物品会不会重复使用

### 8.1 LC416：0/1 背包可行性

数组 `[1,5,11,5]` 的总和是 `22`。问题等价于：能否选择若干数字，恰好凑出 `11`？

定义 `dp[j]`：用已经处理过的数字，能否恰好凑出和 `j`。

```text
dp[0] = True
dp[j] = dp[j] or dp[j - num]
```

![0/1 背包的倒序更新与错误正序对比](/images/dynamic-programming-mastery/03_knapsack_direction.png)

若容量从小到大更新，新得到的 `dp[3]` 会立即参与 `dp[6]`，相当于同一个数字 3 使用了两次。0/1 背包要求每个元素最多一次，所以容量必须倒序。

```python
def canPartition(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2 == 1:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for capacity in range(target, num - 1, -1):
            dp[capacity] |= dp[capacity - num]

    return dp[target]
```

### 8.2 LC322：完全背包最小值

硬币可以重复使用，所以金额从小到大更新：

```python
def coinChange(coins: list[int], amount: int) -> int:
    impossible = amount + 1
    dp = [impossible] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for current in range(coin, amount + 1):
            dp[current] = min(dp[current], dp[current - coin] + 1)

    return -1 if dp[amount] == impossible else dp[amount]
```

背包总口诀：

> **一件一次倒着装，一件多次正着装；可行用或，方案用加，最少用小。**

## 9. 子序列 DP：单串看结尾，双串看前缀

### 9.1 LC300 最长递增子序列

子序列允许跳过元素。定义 `dp[i]` 为**必须以 `nums[i]` 结尾**的最长递增子序列长度：

```python
def lengthOfLIS(nums: list[int]) -> int:
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

返回 `max(dp)` 而不是 `dp[-1]`，因为最优子序列不保证在最后一个位置结束。

### 9.2 LC1143 最长公共子序列

定义 `dp[i][j]` 为 `text1` 前 `i` 个字符与 `text2` 前 `j` 个字符的 LCS 长度。

![最长公共子序列二维状态表](/images/dynamic-programming-mastery/04_lcs_matrix.png)

{% raw %}
$$
dp[i][j]=
\begin{cases}
dp[i-1][j-1]+1,& text1[i-1]=text2[j-1]\\
\max(dp[i-1][j],dp[i][j-1]),& \text{otherwise}
\end{cases}
$$
{% endraw %}

图中的空行和空列代表空字符串。字符相同走左上斜线并加 1；字符不同，从上方和左方取较大值。`abcde` 与 `ace` 的答案为 3。

```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    rows = len(text1) + 1
    columns = len(text2) + 1
    dp = [[0] * columns for _ in range(rows)]

    for i in range(1, rows):
        for j in range(1, columns):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1]
```

口诀：**单串问结尾，双串看前缀；字符同走左上加一，字符异看上左取大。**

## 10. 股票状态机：先画状态，再写公式

每天结束后可以定义：

- `hold`：手里持有一股时的最大利润；
- `cash`：手里没有股票时的最大利润。

以允许多次交易为例：

```text
hold[i] = max(hold[i-1], cash[i-1] - price[i])
cash[i] = max(cash[i-1], hold[i-1] + price[i])
```

每个状态有“今天不操作”的自环；买入让 `cash -> hold`，卖出让 `hold -> cash`。冷冻期或手续费只是增加状态或修改某条边，不应靠死背新公式解决。

口诀：**每天先问持不持；不动沿昨天，动作跨状态；买入减价格，卖出加价格。**

## 11. 初始化与遍历顺序的统一解释

### 11.1 初始化取决于目标运算

| 目标 | 常见初始化 | 原因 |
|---|---|---|
| 方案数 | `dp[0] = 1` | 空方案是递推起点 |
| 最小值 | 其他位置为正无穷 | 不可达状态不能参与 `min` |
| 可行性 | `dp[0] = True` | 什么都不选能凑出 0 |
| 双字符串 | 多开空前缀行列 | 让边界进入统一转移 |

### 11.2 遍历顺序是依赖图的拓扑顺序

- 依赖 `i-1`、`i-2`：从左到右；
- 网格依赖上方、左方：从上到下、从左到右；
- 区间依赖更短区间：先枚举区间长度；
- 0/1 背包压成一维：容量从右到左；
- 完全背包允许重复：容量从左到右。

不是模板规定了顺序，而是当前状态必须在它依赖的状态之后计算。

## 12. 面试白板流程

1. 先说暴力选择：最后一步有哪些可能？
2. 指出相同参数会形成重复子问题。
3. 用完整中文定义状态。
4. 解释每条转移来路为何覆盖全部合法情况。
5. 写初始化与遍历顺序。
6. 用最小例子手推 3 到 5 个状态。
7. 先写清晰完整表，通过后再压缩空间。
8. 报告时间和空间复杂度。

## 13. 高频错误诊断

| 症状 | 根因 | 修复方式 |
|---|---|---|
| 公式会背，换题不会 | 没有从最后一步推导 | 每题先画所有来路 |
| 总差一位 | 混淆前 `i` 个与下标 `i` | 状态句明确“数量”或“下标” |
| 最小值总为 0 | 不可达状态初始化错误 | 使用正无穷并单设 `dp[0]` |
| 背包结果过大 | 0/1 背包用了正序 | 改为倒序并观察本轮覆盖 |
| 返回最后一格错误 | 状态要求“以 i 结尾” | 判断是否应该返回 `max(dp)` |

## 14. 经典题目学习地图

| 类型 | 入门题 | 进阶题 | 识别句 |
|---|---|---|---|
| 线性 DP | LC70、LC198 | LC213、LC53 | 当前如何继承前面的答案 |
| 网格 DP | LC62、LC64 | LC63、LC120 | 当前格从哪里来 |
| 0/1 背包 | LC416 | LC494 | 每件物品最多一次 |
| 完全背包 | LC322 | LC518 | 每种物品可以重复 |
| 单序列 | LC300 | LC647 | 以当前位置结尾或区间 |
| 双序列 | LC1143 | LC72 | 两个前缀之间的答案 |
| 状态机 | LC121、LC122 | LC309、LC714 | 同一天有多种持有状态 |

推荐顺序：

```text
LC70 -> LC198 -> LC62 -> LC416 -> LC322
     -> LC300 -> LC1143 -> LC121 / LC309
```

## 15. 最终记忆口诀

> **重叠子题先记住，状态一句说清楚；最后一步找来路，边界依赖定顺序；一维前看，二维上左；背包次数定正倒，单串结尾双前缀，股票画状态再动手。**

真正掌握 DP 的标准不是能默写某道题，而是看到新约束后，能说明需要修改的是状态、来路、初始化还是遍历方向。先画最小例子，再写代码，公式会变成推理结果，而不再是一串需要硬背的符号。
