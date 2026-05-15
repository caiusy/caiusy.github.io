---
title: ⛰️ LeetCode 150 - 堆/优先队列专题
date: 2026-01-18
updated: 2026-01-18
categories:
  - 算法基础
  - 算法
  -
    - 学习笔记
    - 算法笔记
tags:
  - LeetCode
description: LeetCode 面试 150 题之堆/优先队列专题，含堆结构图解、TopK问题、代码模板
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# ⛰️ 堆/优先队列专题 (5题)

> 🎯 **核心特性**：快速获取最大/最小值，O(log n) 插入删除

---

## 🗺️ 堆的基础知识

```
┌─────────────────────────────────────────────────────────────┐
│                     堆的结构                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  最小堆 (Min Heap)           最大堆 (Max Heap)              │
│       1                           9                         │
│      / \                         / \                        │
│     3   5                       7   8                       │
│    / \                         / \                          │
│   7   8                       3   5                         │
│                                                             │
│  父节点 ≤ 子节点              父节点 ≥ 子节点               │
│  堆顶是最小值                 堆顶是最大值                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Python 中使用 heapq（默认最小堆）                          │
│                                                             │
│  import heapq                                               │
│  heapq.heappush(heap, item)    # 入堆                       │
│  heapq.heappop(heap)           # 出堆                       │
│  heapq.heapify(list)           # 列表转堆                   │
│  heap[0]                       # 查看堆顶                   │
│                                                             │
│  最大堆技巧：存入负数                                        │
│  heapq.heappush(heap, -item)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 堆的常见应用

```
┌─────────────────────────────────────────────────────────────┐
│                 堆的典型应用场景                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Top K 问题                                              │
│     找第K大 → 维护大小为K的最小堆                           │
│     找第K小 → 维护大小为K的最大堆                           │
│                                                             │
│  2. 合并K个有序列表                                         │
│     用堆维护K个列表的当前最小元素                           │
│                                                             │
│  3. 数据流中的中位数                                        │
│     两个堆：最大堆（左半部分）+ 最小堆（右半部分）          │
│                                                             │
│  4. 任务调度                                                │
│     按优先级处理任务                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ LC 215. 数组中的第K个最大元素 🟡

### 题目描述
在未排序的数组中找到第 k 个最大的元素。

### 🎨 图解思路

```
nums = [3,2,1,5,6,4], k = 2

方法1：排序后取第k个 O(n log n)
方法2：最小堆维护k个最大元素 O(n log k)
方法3：快速选择 O(n) 平均

最小堆方法：
维护大小为k的最小堆，堆顶就是第k大

遍历: 3 → [3]
      2 → [2,3]    (k=2)
      1 → [2,3]    (1<2，不入堆)
      5 → [3,5]    (5入堆，弹出2)
      6 → [5,6]    (6入堆，弹出3)
      4 → [5,6]    (4<5，不入堆)

结果: 堆顶 5
```

### 💻 代码实现

```python
import heapq

def findKthLargest(nums: list, k: int) -> int:
    # 方法1：最小堆
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]

    # 方法2：直接用 nlargest
    # return heapq.nlargest(k, nums)[-1]
```

### 🧠 记忆口诀
> **"第K大用最小堆，堆顶就是答案"**

---

## 2️⃣ LC 502. IPO 🔴

### 题目描述
给定若干项目（利润和资本需求），初始资本 w，最多做 k 个项目，求最大资本。

### 🎨 图解思路

```
k = 2, w = 0
profits = [1,2,3]
capitals = [0,1,1]

贪心 + 堆：
每次选择当前资本能做的项目中利润最大的

初始 w=0:
  可做项目: (profit=1, capital=0)
  做项目1 → w=1

w=1:
  可做项目: (profit=2, capital=1), (profit=3, capital=1)
  选最大 → 做项目3 → w=4

结果: 4
```

### 💻 代码实现

```python
import heapq

def findMaximizedCapital(k: int, w: int, profits: list, capital: list) -> int:
    # 按资本排序的项目列表
    projects = sorted(zip(capital, profits))
    
    max_heap = []  # 最大堆存可做项目的利润
    i = 0
    n = len(projects)
    
    for _ in range(k):
        # 把当前资本能做的项目加入堆
        while i < n and projects[i][0] <= w:
            heapq.heappush(max_heap, -projects[i][1])  # 负数模拟最大堆
            i += 1
        
        if not max_heap:
            break
        
        # 选利润最大的项目
        w += -heapq.heappop(max_heap)
    
    return w
```

### 🧠 记忆口诀
> **"资本够就入堆，贪心选最大利润"**

---

## 3️⃣ LC 373. 查找和最小的 K 对数字 🟡

### 题目描述
从两个升序数组中找出和最小的 k 对数字。

### 🎨 图解思路

```
nums1 = [1,7,11], nums2 = [2,4,6], k = 3

可视化矩阵（和）:
      2   4   6
  1   3   5   7
  7   9  11  13
 11  13  15  17

BFS思想：从(0,0)开始，每次扩展右边和下边

初始: (1+2=3, i=0, j=0)
弹出(3), 加入(1+4=5, 0,1) 和 (7+2=9, 1,0)
弹出(5), 加入(1+6=7, 0,2)
弹出(7), ...

结果: [(1,2), (1,4), (1,6)]
```

### 💻 代码实现

```python
import heapq

def kSmallestPairs(nums1: list, nums2: list, k: int) -> list:
    if not nums1 or not nums2:
        return []
    
    result = []
    heap = [(nums1[0] + nums2[0], 0, 0)]  # (sum, i, j)
    visited = {(0, 0)}
    
    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])
        
        # 扩展到右边
        if i + 1 < len(nums1) and (i + 1, j) not in visited:
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
            visited.add((i + 1, j))
        
        # 扩展到下边
        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
            visited.add((i, j + 1))
    
    return result
```

### 🧠 记忆口诀
> **"矩阵BFS，堆取最小扩展"**

---

## 4️⃣ LC 295. 数据流的中位数 🔴

### 题目描述
设计一个支持添加数字和获取中位数的数据结构。

### 🎨 图解思路

```
用两个堆：
- 最大堆 (left): 存较小的一半
- 最小堆 (right): 存较大的一半

保持: len(left) == len(right) 或 len(left) == len(right) + 1

数据流: [1, 2, 3]

add(1): left=[1], right=[]
add(2): left=[1], right=[2]
add(3): left=[1,2], right=[3]

       left(max)    right(min)
          [2]          [3]
          [1]

中位数: 
- 奇数个: left堆顶
- 偶数个: (left堆顶 + right堆顶) / 2
```

### 💻 代码实现

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.left = []   # 最大堆（存负数）
        self.right = []  # 最小堆
    
    def addNum(self, num: int) -> None:
        # 先加入left
        heapq.heappush(self.left, -num)
        
        # 把left最大的给right
        heapq.heappush(self.right, -heapq.heappop(self.left))
        
        # 平衡：保持left >= right
        if len(self.left) < len(self.right):
            heapq.heappush(self.left, -heapq.heappop(self.right))
    
    def findMedian(self) -> float:
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2
```

### 🧠 记忆口诀
> **"左大右小，平衡保持，堆顶找中位"**

---

## 5️⃣ LC 23. 合并 K 个升序链表 🔴

### 题目描述
将 k 个升序链表合并成一个升序链表。

### 🎨 图解思路

```
lists = [[1,4,5], [1,3,4], [2,6]]

用最小堆维护k个链表的当前头节点

初始堆: [1, 1, 2] (三个链表头)

弹出1 → 结果[1]，加入4
堆: [1, 2, 4]

弹出1 → 结果[1,1]，加入3
堆: [2, 3, 4]

...

结果: [1,1,2,3,4,4,5,6]
```

### 💻 代码实现

```python
import heapq

def mergeKLists(lists):
    # 自定义比较（避免链表节点直接比较）
    heap = []
    
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    curr = dummy
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
```

### 🧠 记忆口诀
> **"K个头入堆，弹最小接后继"**

---

## 📊 本章总结

### 题目速查表

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 215 | 第K个最大元素 | 🟡 | Top K |
| 502 | IPO | 🔴 | 贪心+堆 |
| 373 | K对最小和 | 🟡 | 多路归并 |
| 295 | 数据流中位数 | 🔴 | 双堆 |
| 23 | 合并K个链表 | 🔴 | 多路归并 |

### 堆的解题模式

```
┌─────────────────────────────────────────────────────────────┐
│                    堆的解题模式                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Top K 问题:                                                │
│  ─────────                                                  │
│  第K大 → 大小为K的最小堆                                    │
│  第K小 → 大小为K的最大堆                                    │
│                                                             │
│  多路归并:                                                  │
│  ─────────                                                  │
│  堆中维护每路的当前元素                                      │
│  每次取最小，然后加入该路的下一个                           │
│                                                             │
│  双堆技巧:                                                  │
│  ─────────                                                  │
│  中位数：左半最大堆 + 右半最小堆                            │
│  滑动窗口中位数同理                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 🧠 全章记忆口诀

```
TopK问题堆来凑
第K大用小堆守
多路归并维护头
双堆中位数最溜

215 - 第K大元素
502 - IPO项目选择
373 - K对最小和
295 - 数据流中位数
23  - 合并K个链表
```

---

> 📖 **下一篇**：[二分查找专题](/2026/01/18/leetcode-150-binary-search/)

