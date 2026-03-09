---
title: LeetCode #42 接雨水 - 图文详解与费曼总结
date: 2026-03-09 21:56:52
categories:
  - 算法
  - LeetCode
tags:
  - 双指针
  - 动态规划
  - 单调栈
  - 数组
  - Hard
description: 接雨水问题的完整图文解析，包含双指针、动态规划、单调栈三种解法，配有费曼总结和记忆口诀，助你彻底掌握这道经典算法题。
---

# LeetCode #42 接雨水 (Trapping Rain Water)

## 题目描述

给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

**示例 1：**
```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
```

**示例 2：**
```
输入：height = [4,2,0,3,2,5]
输出：9
```

<!-- more -->

## 核心思想图解

### 接雨水的本质

每个位置能接的雨水量 = min(左侧最高柱子, 右侧最高柱子) - 当前柱子高度

```
示例：height = [0,1,0,2,1,0,1,3,2,1,2,1]

步骤1：找到每个位置的左右最高柱子
位置:  0  1  2  3  4  5  6  7  8  9  10 11
高度:  0  1  0  2  1  0  1  3  2  1  2  1
左最高:0  0  1  1  2  2  2  2  3  3  3  3
右最高:3  3  3  3  3  3  3  2  2  2  1  0

步骤2：计算每个位置的接水量
位置 2: min(1,3) - 0 = 1
位置 4: min(2,3) - 1 = 1
位置 5: min(2,3) - 0 = 2
位置 6: min(2,3) - 1 = 1
位置 9: min(3,2) - 1 = 1
总计: 1+1+2+1+1 = 6
```

### 视觉图示

```
高度图：
    █
█   █ █   █
█ █ █ █ █ █ █ █
0 1 0 2 1 0 1 3 2 1 2 1

接水后：
    █
█ ≈ █≈█ ≈ █
█≈█≈█≈█≈█≈█≈█≈█
0 1 0 2 1 0 1 3 2 1 2 1
(≈ 表示水)
```

## 解法一：双指针法（最优解）

### 核心思路

使用左右双指针，维护左右两侧的最大高度，从两端向中间移动。

**关键洞察**：
- 如果 `left_max < right_max`，则左指针位置的接水量由 `left_max` 决定
- 如果 `right_max < left_max`，则右指针位置的接水量由 `right_max` 决定

### 算法流程

```mermaid
graph TD
    A[初始化 left=0, right=n-1] --> B[left_max=0, right_max=0]
    B --> C{left < right?}
    C -->|否| D[返回 result]
    C -->|是| E{height[left] < height[right]?}
    E -->|是| F[更新 left_max]
    F --> G[计算左侧接水量]
    G --> H[left++]
    H --> C
    E -->|否| I[更新 right_max]
    I --> J[计算右侧接水量]
    J --> K[right--]
    K --> C
```

### 代码实现

**Python:**
```python
def trap(height: List[int]) -> int:
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    result = 0

    while left < right:
        if height[left] < height[right]:
            # 左侧较低，处理左侧
            if height[left] >= left_max:
                left_max = height[left]
            else:
                result += left_max - height[left]
            left += 1
        else:
            # 右侧较低，处理右侧
            if height[right] >= right_max:
                right_max = height[right]
            else:
                result += right_max - height[right]
            right -= 1

    return result
```

**Java:**
```java
public int trap(int[] height) {
    if (height == null || height.length == 0) {
        return 0;
    }

    int left = 0, right = height.length - 1;
    int leftMax = 0, rightMax = 0;
    int result = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                result += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                result += rightMax - height[right];
            }
            right--;
        }
    }

    return result;
}
```

### 复杂度分析
- **时间复杂度**：O(n)，只需遍历一次数组
- **空间复杂度**：O(1)，只使用常数额外空间

---

## 解法二：动态规划法

### 核心思路

预先计算每个位置的左侧最大值和右侧最大值，然后遍历计算接水量。

### 代码实现

**Python:**
```python
def trap(height: List[int]) -> int:
    if not height:
        return 0

    n = len(height)
    left_max = [0] * n
    right_max = [0] * n

    # 计算每个位置的左侧最大值
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])

    # 计算每个位置的右侧最大值
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])

    # 计算接水量
    result = 0
    for i in range(n):
        result += min(left_max[i], right_max[i]) - height[i]

    return result
```

### 复杂度分析
- **时间复杂度**：O(n)，需要三次遍历
- **空间复杂度**：O(n)，需要两个辅助数组

---

## 解法三：单调栈法

### 核心思路

使用单调递减栈，当遇到比栈顶高的柱子时，说明可以形成凹槽接水。

### 代码实现

**Python:**
```python
def trap(height: List[int]) -> int:
    stack = []
    result = 0

    for i in range(len(height)):
        while stack and height[i] > height[stack[-1]]:
            top = stack.pop()
            if not stack:
                break

            # 计算宽度和高度
            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[top]
            result += distance * bounded_height

        stack.append(i)

    return result
```

### 复杂度分析
- **时间复杂度**：O(n)，每个元素最多入栈出栈一次
- **空间复杂度**：O(n)，栈的空间

---

## 🎯 费曼总结

### 用最简单的话解释接雨水

想象你站在一排高低不同的柱子前，下雨了。**每个位置能接多少水，取决于它左右两边最高的柱子中较矮的那个**。

就像一个木桶，水位由最短的那块板决定。对于每个位置：
1. 找到左边最高的柱子
2. 找到右边最高的柱子
3. 取两者中较小的值
4. 减去当前位置的高度，就是这个位置能接的水

### 为什么双指针法最优？

动态规划需要先遍历两遍数组记录左右最大值，再遍历一遍计算结果。

双指针法的巧妙之处：**我们不需要知道所有位置的左右最大值，只需要知道当前位置的接水量由哪一侧决定**。

- 如果左边的最大值 < 右边的最大值，那么左指针位置的接水量只由左边决定
- 反之，右指针位置的接水量只由右边决定

这样我们只需要一次遍历，用 O(1) 空间就能解决问题。

---

## 📝 记忆口诀

```
接雨水，看两边，
左右最高取最小，
减去当前柱子高，
就是此处水能装。

双指针，最优解，
谁矮处理谁那边，
一次遍历空间省，
时间空间都最优。
```

**核心记忆点：**
- 🪣 **木桶原理**：水位由短板决定 → min(左最高, 右最高)
- 👈👉 **双指针策略**：谁矮处理谁 → 矮的一侧决定接水量
- 📊 **接水公式**：min(left_max, right_max) - height[i]

---

## 📊 三种解法对比

| 解法 | 时间复杂度 | 空间复杂度 | 优点 | 缺点 |
|------|-----------|-----------|------|------|
| 双指针 | O(n) | O(1) | 最优解，空间最省 | 理解稍难 |
| 动态规划 | O(n) | O(n) | 思路直观，易理解 | 需要额外数组 |
| 单调栈 | O(n) | O(n) | 按层计算，思路独特 | 代码较复杂 |

**推荐顺序**：
1. 先理解动态规划（最直观）
2. 再优化到双指针（面试首选）
3. 最后了解单调栈（拓展思路）

---

## 🎓 易错点与注意事项

### 易错点 1：边界条件
```python
# ❌ 错误：没有检查空数组
def trap(height):
    left, right = 0, len(height) - 1  # 空数组会报错

# ✅ 正确：先检查边界
def trap(height):
    if not height or len(height) < 3:
        return 0
```

### 易错点 2：双指针移动条件
```python
# ❌ 错误：使用 <= 会导致重复计算
while left <= right:

# ✅ 正确：使用 < 避免重复
while left < right:
```

### 易错点 3：最大值更新时机
```python
# 关键：先判断是否需要更新最大值，再计算接水量
if height[left] >= left_max:
    left_max = height[left]  # 更新最大值，此位置不接水
else:
    result += left_max - height[left]  # 计算接水量
```

---

## 🔗 相关题目

- [LeetCode #11 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/) - 双指针基础题
- [LeetCode #84 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/) - 单调栈应用
- [LeetCode #407 接雨水 II](https://leetcode.cn/problems/trapping-rain-water-ii/) - 二维版本，使用优先队列

---

## 💡 总结

接雨水是一道经典的双指针题目，核心在于理解：
1. **木桶原理**：每个位置的接水量由两侧较矮的一侧决定
2. **双指针优化**：不需要预先计算所有位置的左右最大值
3. **贪心思想**：谁矮处理谁，保证当前决策最优

掌握这道题，你就掌握了双指针、动态规划、单调栈三种重要算法思想的应用。

---

**难度**：⭐⭐⭐ Hard
**标签**：`数组` `双指针` `动态规划` `单调栈`
**推荐指数**：⭐⭐⭐⭐⭐（必刷题）
