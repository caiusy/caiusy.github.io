---
title: 🔢 LeetCode 150 - 回溯算法专题
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
description: LeetCode 面试 150 题之回溯算法专题，含决策树图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 🔄 回溯算法专题 (7题)

> 🎯 **核心思想**：尝试所有可能，走不通就回头

---

## 🗺️ 回溯算法的本质

```
┌─────────────────────────────────────────────────────────────┐
│                   回溯 = 决策树的遍历                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                        []                                   │
│                    /   |   \                                │
│                  [1]  [2]  [3]     ← 第一层决策              │
│                 / \    |                                    │
│             [1,2][1,3][2,3]        ← 第二层决策              │
│               |                                             │
│            [1,2,3]                 ← 第三层决策              │
│                                                             │
│  回溯三要素：                                                │
│  1. 路径：已经做出的选择                                     │
│  2. 选择列表：当前可以做的选择                               │
│  3. 结束条件：到达决策树底层                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 回溯算法模板

```python
def backtrack(path, choices):
    # 结束条件
    if 满足结束条件:
        result.append(path[:])  # 注意拷贝！
        return
    
    for choice in choices:
        # 1. 做选择
        path.append(choice)
        
        # 2. 递归进入下一层决策
        backtrack(path, new_choices)
        
        # 3. 撤销选择（回溯）
        path.pop()
```

### 🧠 回溯口诀
> **"选择、递归、撤销"** —— 回溯三部曲

---

## 1️⃣ LC 17. 电话号码的字母组合 🟡

### 题目描述
给定电话号码，返回所有可能的字母组合。

### 🎨 图解思路

```
digits = "23"

2 → "abc"
3 → "def"

决策树:
           ""
       /   |   \
      a    b    c      ← 选择2对应的字母
     /|\  /|\  /|\
    d e f d e f d e f  ← 选择3对应的字母

结果: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

### 💻 代码实现

```python
def letterCombinations(digits: str) -> list:
    if not digits:
        return []
    
    phone = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, path):
        if index == len(digits):
            result.append(''.join(path))
            return
        
        for char in phone[digits[index]]:
            path.append(char)
            backtrack(index + 1, path)
            path.pop()
    
    backtrack(0, [])
    return result
```

### 🧠 记忆口诀
> **"每个数字选一个字母"**

---

## 2️⃣ LC 77. 组合 🟡

### 题目描述
从 1 到 n 中选择 k 个数的所有组合。

### 🎨 图解思路

```
n = 4, k = 2

决策树（每次只能选比自己大的数，避免重复）:
              []
         /  |  |  \
       [1] [2] [3] [4]
      / | \  |
  [1,2][1,3][1,4] [2,3][2,4] [3,4]

结果: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

### 💻 代码实现

```python
def combine(n: int, k: int) -> list:
    result = []
    
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        
        # 剪枝：剩余元素不够用了
        if k - len(path) > n - start + 1:
            return
        
        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()
    
    backtrack(1, [])
    return result
```

### 🧠 记忆口诀
> **"从start开始选，选够k个停"**

---

## 3️⃣ LC 46. 全排列 🟡

### 题目描述
返回数组的所有排列。

### 🎨 图解思路

```
nums = [1, 2, 3]

决策树（每个数只能用一次）:
                    []
            /       |        \
          [1]      [2]       [3]
         /   \    /   \     /   \
      [1,2] [1,3] [2,1] [2,3] [3,1] [3,2]
        |     |     |     |     |     |
    [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]
```

### 💻 代码实现

```python
def permute(nums: list) -> list:
    result = []
    
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            
            path.append(nums[i])
            used[i] = True
            
            backtrack(path, used)
            
            path.pop()
            used[i] = False
    
    backtrack([], [False] * len(nums))
    return result
```

### 🧠 记忆口诀
> **"用过的标记，没用过的都能选"**

---

## 4️⃣ LC 39. 组合总和 🟡

### 题目描述
找出所有和为 target 的组合（数字可以重复使用）。

### 🎨 图解思路

```
candidates = [2, 3, 6, 7], target = 7

决策树:
                    []
         /     |      \      \
       [2]    [3]    [6]    [7] ✓
      / | \    |      |
   [2,2][2,3][2,6] [3,3]  [6,?]
    /|\   |
[2,2,2][2,2,3]✓ [2,3,?]
  |
[2,2,2,?] 超过7，剪枝
```

### 💻 代码实现

```python
def combinationSum(candidates: list, target: int) -> list:
    result = []
    candidates.sort()  # 排序便于剪枝
    
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # 剪枝
            
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i不是i+1，可重复
            path.pop()
    
    backtrack(0, [], target)
    return result
```

### 🧠 记忆口诀
> **"可以重复选，但只能往后选"**

---

## 5️⃣ LC 52. N 皇后 II 🔴

### 题目描述
返回 N 皇后问题的解的数量。

### 🎨 图解思路

```
n = 4

一个有效解:
. Q . .
. . . Q
Q . . .
. . Q .

约束条件：
1. 每行只能放一个皇后
2. 每列只能放一个皇后
3. 每条对角线只能放一个皇后

对角线编号技巧:
- 主对角线 (\): row - col 相同
- 副对角线 (/): row + col 相同
```

### 💻 代码实现

```python
def totalNQueens(n: int) -> int:
    count = 0
    cols = set()       # 列冲突
    diag1 = set()      # 主对角线 (row - col)
    diag2 = set()      # 副对角线 (row + col)
    
    def backtrack(row):
        nonlocal count
        
        if row == n:
            count += 1
            return
        
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            backtrack(row + 1)
            
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    backtrack(0)
    return count
```

### 🧠 记忆口诀
> **"列和两条对角线，都不能冲突"**

---

## 6️⃣ LC 22. 括号生成 🟡

### 题目描述
生成 n 对有效的括号组合。

### 🎨 图解思路

```
n = 2

决策树（剪枝条件：右括号不能多于左括号）:
                ""
               /
              (
            /   \
          ((    ()
          |    /
         (()  ()(
          |    |
        (()) ()()

结果: ["(())", "()()"]
```

### 💻 代码实现

```python
def generateParenthesis(n: int) -> list:
    result = []
    
    def backtrack(path, left, right):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return
        
        if left < n:
            path.append('(')
            backtrack(path, left + 1, right)
            path.pop()
        
        if right < left:
            path.append(')')
            backtrack(path, left, right + 1)
            path.pop()
    
    backtrack([], 0, 0)
    return result
```

### 🧠 记忆口诀
> **"左括号随时加，右括号不超左"**

---

## 7️⃣ LC 79. 单词搜索 🟡

### 题目描述
在二维网格中搜索单词。

### 🎨 图解思路

```
board:
A B C E
S F C S
A D E E

word = "ABCCED"

从 A 开始，DFS + 回溯:
A → B → C → C → E → D ✓
```

### 💻 代码实现

```python
def exist(board: list, word: str) -> bool:
    m, n = len(board), len(board[0])
    
    def backtrack(i, j, k):
        if k == len(word):
            return True
        
        if i < 0 or i >= m or j < 0 or j >= n:
            return False
        
        if board[i][j] != word[k]:
            return False
        
        # 标记已访问
        temp = board[i][j]
        board[i][j] = '#'
        
        # 四个方向搜索
        found = (backtrack(i + 1, j, k + 1) or
                 backtrack(i - 1, j, k + 1) or
                 backtrack(i, j + 1, k + 1) or
                 backtrack(i, j - 1, k + 1))
        
        # 恢复
        board[i][j] = temp
        
        return found
    
    for i in range(m):
        for j in range(n):
            if backtrack(i, j, 0):
                return True
    
    return False
```

### 🧠 记忆口诀
> **"DFS四方向，访问要标记"**

---

## 📊 本章总结

### 回溯问题分类

| 类型 | 特点 | 典型题目 |
|------|------|----------|
| 组合问题 | 不考虑顺序 | 77, 39 |
| 排列问题 | 考虑顺序 | 46 |
| 子集问题 | 所有可能 | 78 |
| 搜索问题 | 在空间中找路径 | 79 |
| 棋盘问题 | 放置约束 | 52 |
| 括号问题 | 合法性约束 | 22 |

### 回溯 vs 动态规划

```
┌──────────────┬──────────────────────────────┐
│    回溯      │         动态规划             │
├──────────────┼──────────────────────────────┤
│ 找所有解     │ 找最优解/计数                │
│ 暴力穷举     │ 记忆化避免重复               │
│ 时间换空间   │ 空间换时间                   │
└──────────────┴──────────────────────────────┘
```

### 🧠 全章记忆口诀

```
电话组合全排列
组合总和皇后解
括号生成单词找
回溯七题全拿下

电话 - 电话号码的字母组合 (17)
组合 - 组合 (77)
全排列 - 全排列 (46)
组合总和 - 组合总和 (39)
皇后 - N皇后 II (52)
括号 - 括号生成 (22)
单词 - 单词搜索 (79)
```

---

> 📖 **返回**：[LeetCode 150 题总目录](/2026/01/18/leetcode-150-index/)

