---
title: 📦 LeetCode 150 - 栈与队列专题
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
description: LeetCode 面试 150 题之栈与队列专题，含单调栈图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 📦 栈与队列专题 (7题)

> 🎯 **核心特性**：栈 LIFO（后进先出），队列 FIFO（先进先出）

---

## 🗺️ 栈的核心应用场景

```
┌─────────────────────────────────────────────────────────────┐
│                    栈的应用场景                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 括号匹配                                                │
│     遇到左括号入栈，遇到右括号出栈匹配                       │
│                                                             │
│  2. 表达式求值                                               │
│     操作数栈 + 运算符栈                                      │
│                                                             │
│  3. 单调栈                                                   │
│     找下一个更大/更小元素                                    │
│                                                             │
│  4. 路径简化                                                 │
│     处理 "." 和 ".."                                        │
│                                                             │
│       ┌───┐                                                 │
│       │ C │ ← Top (后进先出)                                │
│       ├───┤                                                 │
│       │ B │                                                 │
│       ├───┤                                                 │
│       │ A │                                                 │
│       └───┘                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1️⃣ LC 20. 有效的括号 🟢

### 题目描述
判断括号字符串是否有效。

### 🎨 图解思路

```
s = "([{}])"

处理过程:
字符    操作      栈状态
(      入栈      [(]
[      入栈      [(, []
{      入栈      [(, [, {]
}      出栈匹配   [(, []      ✓ { 匹配
]      出栈匹配   [(]         ✓ [ 匹配
)      出栈匹配   []          ✓ ( 匹配

栈为空 → 有效！
```

### 💻 代码实现

```python
def isValid(s: str) -> bool:
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    
    for char in s:
        if char in mapping:
            # 右括号：出栈匹配
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            # 左括号：入栈
            stack.append(char)
    
    return len(stack) == 0
```

### 🧠 记忆口诀
> **"左入右出，空栈有效"**

---

## 2️⃣ LC 71. 简化路径 🟡

### 题目描述
简化 Unix 风格的绝对路径。

### 🎨 图解思路

```
path = "/a/./b/../../c/"

处理规则:
.   → 当前目录，忽略
..  → 上级目录，出栈
其他 → 目录名，入栈

处理过程:
a   → 入栈 → [a]
.   → 忽略 → [a]
b   → 入栈 → [a, b]
..  → 出栈 → [a]
..  → 出栈 → []
c   → 入栈 → [c]

结果: "/c"
```

### 💻 代码实现

```python
def simplifyPath(path: str) -> str:
    stack = []
    
    for part in path.split('/'):
        if part == '..':
            if stack:
                stack.pop()
        elif part and part != '.':
            stack.append(part)
    
    return '/' + '/'.join(stack)
```

### 🧠 记忆口诀
> **"点忽略，双点出栈，其他入栈"**

---

## 3️⃣ LC 155. 最小栈 🟡

### 题目描述
设计一个支持 O(1) 获取最小值的栈。

### 🎨 图解思路

```
使用辅助栈同步记录当前最小值

主栈      辅助栈（记录当前最小）
push(5)   [5]       [5]
push(3)   [5,3]     [5,3]      ← 3更小
push(7)   [5,3,7]   [5,3,3]    ← 最小仍是3
pop()     [5,3]     [5,3]
getMin()  返回 3
```

### 💻 代码实现

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        self.stack.append(val)
        # 辅助栈：记录当前最小值
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
        else:
            self.min_stack.append(self.min_stack[-1])
    
    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self) -> int:
        return self.stack[-1]
    
    def getMin(self) -> int:
        return self.min_stack[-1]
```

### 🧠 记忆口诀
> **"辅助栈同步记录最小值"**

---

## 4️⃣ LC 150. 逆波兰表达式求值 🟡

### 题目描述
计算逆波兰表达式（后缀表达式）的值。

### 🎨 图解思路

```
tokens = ["2","1","+","3","*"]

等价于: (2 + 1) * 3 = 9

处理过程（遇到运算符弹出两个操作数）:
2   → 入栈 → [2]
1   → 入栈 → [2, 1]
+   → 弹出1,2，计算2+1=3 → [3]
3   → 入栈 → [3, 3]
*   → 弹出3,3，计算3*3=9 → [9]

结果: 9
```

### 💻 代码实现

```python
def evalRPN(tokens: list) -> int:
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b, a = stack.pop(), stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                # 注意：Python除法向零取整
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]
```

### 🧠 记忆口诀
> **"数字入栈，运算符弹两个算"**

---

## 5️⃣ LC 224. 基本计算器 🔴

### 题目描述
实现一个基本的计算器（支持 +、-、括号）。

### 🎨 图解思路

```
s = "1 + (2 - 3)"

使用栈保存括号外的状态

遇到 ( → 保存当前 result 和 sign，重置
遇到 ) → 恢复之前的状态并累加

处理: 1 + (2 - 3)
1      → result = 1
+      → sign = 1
(      → 保存(1, 1)，重置 result=0
2      → result = 2
-      → sign = -1
3      → result = 2 + (-1)*3 = -1
)      → result = 1 + 1*(-1) = 0
```

### 💻 代码实现

```python
def calculate(s: str) -> int:
    stack = []
    result = 0
    num = 0
    sign = 1
    
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            # 保存当前状态
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            # 恢复状态
            result = result * stack.pop() + stack.pop()
    
    return result + sign * num
```

### 🧠 记忆口诀
> **"括号保存状态，出来恢复累加"**

---

## 6️⃣ LC 227. 基本计算器 II 🟡

### 题目描述
实现计算器（支持 +、-、*、/，无括号）。

### 🎨 图解思路

```
s = "3+2*2"

* / 优先级高于 + -

策略：用栈保存待加的数
遇到 + 入栈正数
遇到 - 入栈负数
遇到 * / 弹出栈顶计算后入栈

处理: 3 + 2 * 2
3   → 栈 [3]
+   → 记录 op = +
2   → 栈 [3, 2]
*   → 记录 op = *
2   → 弹出2，计算2*2=4，栈 [3, 4]

结果: sum([3, 4]) = 7
```

### 💻 代码实现

```python
def calculate(s: str) -> int:
    stack = []
    num = 0
    op = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        if char in '+-*/' or i == len(s) - 1:
            if op == '+':
                stack.append(num)
            elif op == '-':
                stack.append(-num)
            elif op == '*':
                stack.append(stack.pop() * num)
            elif op == '/':
                stack.append(int(stack.pop() / num))
            
            op = char
            num = 0
    
    return sum(stack)
```

### 🧠 记忆口诀
> **"加减入栈，乘除先算"**

---

## 7️⃣ LC 772. 基本计算器 III 🔴

### 题目描述
实现完整计算器（支持 +、-、*、/、括号）。

### 🎨 图解思路

```
结合 224 和 227 的思路

方法1：递归处理括号
方法2：双栈（操作数栈 + 运算符栈）
```

### 💻 代码实现（递归）

```python
def calculate(s: str) -> int:
    def helper(s, start):
        stack = []
        num = 0
        op = '+'
        i = start
        
        while i < len(s):
            char = s[i]
            
            if char.isdigit():
                num = num * 10 + int(char)
            
            if char == '(':
                num, i = helper(s, i + 1)
            
            if char in '+-*/)' or i == len(s) - 1:
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                elif op == '/':
                    stack.append(int(stack.pop() / num))
                
                if char == ')':
                    return sum(stack), i
                
                op = char
                num = 0
            
            i += 1
        
        return sum(stack), i
    
    return helper(s, 0)[0]
```

### 🧠 记忆口诀
> **"遇括号递归，其他同227"**

---

## 📊 本章总结

### 栈的应用模式

| 模式 | 核心思想 | 典型题目 |
|------|----------|----------|
| 括号匹配 | 左入右出 | 20 |
| 路径处理 | 目录入栈，..出栈 | 71 |
| 辅助栈 | 同步维护额外信息 | 155 |
| 表达式求值 | 操作数栈 + 运算符处理 | 150, 224, 227 |

### 单调栈模板（补充）

```python
def monotonic_stack(nums):
    """找每个元素右边第一个更大的元素"""
    stack = []  # 存索引
    result = [-1] * len(nums)
    
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result
```

### 🧠 全章记忆口诀

```
括号路径最小栈
逆波兰后计算器
三道计算器升级
栈的七题记心里

括号 - 有效的括号 (20)
路径 - 简化路径 (71)
最小栈 - 最小栈 (155)
逆波兰 - 逆波兰表达式求值 (150)
计算器 - 基本计算器系列 (224, 227, 772)
```

---

> 📖 **下一篇**：[堆/优先队列专题](/2026/01/18/leetcode-150-heap/)

