---
title: 单调栈：从 O(n²) 到 O(n) 的状态压缩思路
date: 2026-02-19 17:00:00
tags:
  - 算法
categories:
  - 算法基础
  - 算法
  -
    - 学习笔记
    - 算法笔记
description: 深度解析单调栈的原理、应用和实战技巧，用费曼学习法帮你彻底理解这个强大的算法工具
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 单调栈完全指南：从 O(n²) 到 O(n) 的优雅跃迁

## 🎯 一句话本质

> **单调栈通过维护单调性来及时淘汰无用候选元素，将"寻找下一个更大/更小元素"问题从 O(n²) 优化到 O(n)。**

它不是一种数据结构，而是一种**遍历策略** —— 通过栈的单调性保证"不漏掉答案"的同时"跳过不可能的区域"。

---

## 🤔 为什么需要单调栈？

### 问题场景

给定数组 `[2, 1, 5, 6, 2, 3]`，对于每个元素，找到它右边第一个比它大的元素。

**暴力解法**：
```python
def next_greater_brute(nums):
    n = len(nums)
    result = [-1] * n
    for i in range(n):
        for j in range(i+1, n):  # 向右扫描
            if nums[j] > nums[i]:
                result[i] = nums[j]
                break
    return result
```

- **时间复杂度**：O(n²) - 每个元素都要向右扫描
- **问题**：大量重复计算

### 💡 关键洞察

**如果 `nums[i] < nums[j]`（i < j），那么对于 j 右边的元素来说，i 永远不可能是答案！**

**为什么？** 因为 j 更大且更靠右，它会"遮挡"住 i。

这就是单调栈的核心：**维护一个递减的候选序列，及时淘汰无用元素**。

---

## 🔬 工作原理：数据流与状态转移

### 单调栈的处理流程

以 `[2, 1, 5, 6, 2, 3]` 为例，维护一个**单调递减栈**（栈底到栈顶递减）：

| 步骤 | 当前元素 | 栈状态（存索引） | 操作 | 发现的答案 |
|------|----------|------------------|------|------------|
| 1 | 2 (i=0) | [0] | 直接入栈 | - |
| 2 | 1 (i=1) | [0, 1] | 1 < 2，直接入栈 | - |
| 3 | 5 (i=2) | [2] | 5 > 1，弹出1；5 > 2，弹出2；5入栈 | nums[1]=5, nums[0]=5 |
| 4 | 6 (i=3) | [3] | 6 > 5，弹出5；6入栈 | nums[2]=6 |
| 5 | 2 (i=4) | [3, 4] | 2 < 6，直接入栈 | - |
| 6 | 3 (i=5) | [3, 5] | 3 > 2，弹出2；3 < 6，3入栈 | nums[4]=3 |

**最终结果**：`[5, 5, 6, -1, 3, -1]`

### 📊 可视化：栈的动态演变

![单调栈状态变化](/images/monotonic-stack/monotonic_stack_states.png)

上图展示了处理数组的6个关键步骤，每个步骤包含：
- **输入数组**：当前处理的元素用红色高亮
- **栈状态**：绿色方块表示栈中的元素
- **结果数组**：黄色表示已找到答案，灰色表示未找到

### 🎬 动画演示

![单调栈动画](/images/monotonic-stack/monotonic_stack_animation.gif)

动画展示了完整的处理流程，可以清晰看到：
- 元素逐个进入处理
- 栈的动态变化（入栈/出栈）
- 结果数组的实时更新


### 💻 核心代码实现

```python
def next_greater_element(nums):
    n = len(nums)
    result = [-1] * n  # 初始化答案数组
    stack = []  # 单调栈（存储索引）

    for i in range(n):
        # 当前元素 > 栈顶元素时，找到了栈顶的答案
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]  # 记录答案
        stack.append(i)  # 当前索引入栈

    return result
```

**数据流向**：
```
输入数组 → 逐个处理 → 单调栈（维护候选） → 输出答案数组
         ↑                    ↓
         └────── 弹出时记录答案 ──┘
```

---

## ⏱️ 时间复杂度证明

![时间复杂度可视化](/images/monotonic-stack/monotonic_stack_complexity.png)

### 为什么是 O(n)？

**关键洞察**：每个元素最多入栈一次，出栈一次。

**证明**：
- 外层循环：遍历 n 个元素 → O(n)
- 内层 while 循环：看似嵌套，但总共最多弹出 n 次
  - 每个元素入栈 1 次
  - 每个元素出栈 ≤ 1 次
  - 总操作次数 ≤ 2n

**均摊分析**：
```
总入栈次数 = n
总出栈次数 ≤ n
总操作 = n + n = 2n = O(n)
```

---

## 🎯 经典问题实战

### 问题 1：下一个更大元素 II（循环数组）

**问题**：数组是循环的，如 `[1, 2, 1]` 中，最后一个 1 的答案是 2。

**解法**：将数组复制一遍，模拟循环。

```python
def next_greater_circular(nums):
    n = len(nums)
    result = [-1] * n
    stack = []

    # 遍历两遍数组（模拟循环）
    for i in range(2 * n):
        idx = i % n  # 实际索引
        while stack and nums[idx] > nums[stack[-1]]:
            result[stack.pop()] = nums[idx]
        if i < n:  # 只在第一遍时入栈
            stack.append(idx)

    return result
```


### 问题 2：柱状图中最大的矩形

**问题**：给定柱状图高度 `[2, 1, 5, 6, 2, 3]`，找最大矩形面积。

**核心思路**：对于每个柱子，找到它左右两边第一个比它矮的柱子，计算以它为高的矩形面积。

```python
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    heights = [0] + heights + [0]  # 哨兵技巧

    for i, h in enumerate(heights):
        while stack and h < heights[stack[-1]]:
            height_idx = stack.pop()
            height = heights[height_idx]
            width = i - stack[-1] - 1  # 左右边界
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
```

![柱状图最大矩形](/images/monotonic-stack/monotonic_stack_rectangle.png)

上图展示了柱状图中最大矩形的可视化，红色虚线框标出了面积为10的最大矩形。

### 问题 3：每日温度

**问题**：给定温度数组 `[73, 74, 75, 71, 69, 72, 76, 73]`，计算每天需要等几天才能等到更暖和的温度。

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []

    for i, temp in enumerate(temperatures):
        while stack and temp > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx  # 天数差
        stack.append(i)

    return result
```

**输出**：`[1, 1, 4, 2, 1, 1, 0, 0]`


---

## 🧠 费曼总结：用简单的话解释单调栈

### 给10岁小孩的解释

想象你在排队买票，队伍里每个人都有一个身高牌。你的任务是：**告诉每个人，他后面第一个比他高的人是谁**。

**笨办法**：每个人都要回头看一遍后面所有人 → 很慢（O(n²)）

**聪明办法（单调栈）**：
1. 维护一个"候选队列"，队列里的人从前到后越来越矮
2. 新人来了，如果他比队尾的人高：
   - 队尾的人找到答案了！（就是这个新人）
   - 把队尾的人踢出去（他已经没用了）
   - 继续比较，直到新人不比队尾高
3. 新人加入队尾

**为什么这样快？** 每个人最多进队一次、出队一次 → O(n)

### 核心要点

1. **什么时候用单调栈？**
   - 需要找"下一个更大/更小元素"
   - 需要找"左右边界"
   - 暴力解法需要嵌套循环

2. **单调栈的本质**
   - 维护一个候选序列
   - 新元素到来时，淘汰被"遮挡"的元素
   - 弹出时记录答案

3. **为什么是 O(n)？**
   - 每个元素最多入栈一次，出栈一次
   - 总操作次数 ≤ 2n

4. **单调递增 vs 单调递减**
   - 单调递增栈：找下一个更小元素
   - 单调递减栈：找下一个更大元素

### 记忆口诀

```
遇到"下一个更大"，单调栈来帮忙
栈顶小于当前值，弹出记录答案忙
每个元素进出一次，时间复杂度 O(n) 强
```


---

## 🧠 费曼总结：用简单的话解释单调栈

### 给10岁小孩解释

想象你在排队买冰淇淋，队伍里每个人都有不同的身高。

**问题**：每个人都想知道，自己后面第一个比自己高的人是谁？

**笨办法**：每个人都回头看，一个一个找 → 很慢（O(n²)）

**聪明办法（单调栈）**：
1. 维护一个"候选队列"，队列里的人从前到后越来越矮
2. 新人来了：
   - 如果比队尾的人矮 → 直接加入队列
   - 如果比队尾的人高 → 队尾的人找到答案了！把他踢出去，继续比较
3. 每个人最多进出队列一次 → 很快（O(n)）

### 核心要点

1. **什么时候用单调栈？**
   - 需要找"下一个更大/更小元素"
   - 需要找"左右边界"
   - 暴力解法需要嵌套循环

2. **单调栈的本质**
   - 维护一个候选序列
   - 新元素到来时，淘汰被"遮挡"的元素
   - 弹出时记录答案

3. **为什么是 O(n)？**
   - 每个元素最多入栈一次，出栈一次
   - 总操作次数 ≤ 2n

4. **单调递增 vs 单调递减**
   - 单调递增栈：找下一个更小元素
   - 单调递减栈：找下一个更大元素

### 记忆口诀

```
遇到"下一个更大"，单调栈来帮忙
栈顶小于当前值，弹出记录答案忙
每个元素进出一次，时间复杂度 O(n) 强
```


---

## 📝 代码模板

### 找下一个更大元素（单调递减栈）

```python
def next_greater(nums):
    result = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
```

### 找下一个更小元素（单调递增栈）

```python
def next_smaller(nums):
    result = [-1] * len(nums)
    stack = []
    for i in range(len(nums)):
        while stack and nums[i] < nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result
```


---

## 📚 LeetCode 题单

### 基础题（Easy）
- [496. 下一个更大元素 I](https://leetcode.com/problems/next-greater-element-i/)
- [739. 每日温度](https://leetcode.com/problems/daily-temperatures/)

### 进阶题（Medium）
- [503. 下一个更大元素 II](https://leetcode.com/problems/next-greater-element-ii/)
- [901. 股票价格跨度](https://leetcode.com/problems/online-stock-span/)

### 困难题（Hard）
- [84. 柱状图中最大的矩形](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [42. 接雨水](https://leetcode.com/problems/trapping-rain-water/)
- [85. 最大矩形](https://leetcode.com/problems/maximal-rectangle/)

---

## 🎓 进阶思考

### 1. 单调队列 vs 单调栈
- **单调队列**：支持队首队尾操作，用于滑动窗口最大值
- **单调栈**：只支持栈顶操作，用于寻找边界

### 2. 双单调栈
某些问题需要同时维护左右边界，可以用两个单调栈分别处理。

### 3. 单调栈 + DP
在某些 DP 问题中，单调栈可以优化状态转移。

---

## 🔑 关键要点总结

| 特性 | 说明 |
|------|------|
| **适用场景** | 找下一个更大/更小元素、左右边界 |
| **时间复杂度** | O(n) |
| **空间复杂度** | O(n) |
| **核心思想** | 维护单调性，及时淘汰无用元素 |
| **存储内容** | 通常存索引而非值 |

---

**创建时间**：2026-02-19  
**标签**：#算法 #单调栈 #数据结构 #优化技巧

