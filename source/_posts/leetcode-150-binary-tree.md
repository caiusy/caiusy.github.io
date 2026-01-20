---
title: 🌳 LeetCode 150 - 二叉树专题
date: 2026-01-18
updated: 2026-01-18
categories:
  - 算法
  - LeetCode
  - 算法
  - LeetCode
tags: LeetCode
  - LeetCode
  - leetcode面试150
  - 二叉树
  - Binary Tree
  - 面试
description: LeetCode 面试 150 题之二叉树专题，含遍历图解、递归思维、代码模板
---

# 🌳 二叉树专题 (14题)

> 🎯 **核心思想**：递归思维 + 分解问题

---

## 🗺️ 二叉树的思维模式

```
┌─────────────────────────────────────────────────────────────┐
│                 二叉树的两种思维模式                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  模式一：遍历思维                                            │
│  ────────────────                                           │
│  用一个 traverse 函数遍历整棵树                              │
│  在遍历过程中更新外部变量                                    │
│                                                             │
│  模式二：分解问题思维                                        │
│  ──────────────────                                         │
│  将问题分解为子问题                                          │
│  通过子问题的答案推导出原问题的答案                          │
│                                                             │
│            1                                                │
│           / \                                               │
│          2   3        问题(根) = f(问题(左), 问题(右))       │
│         / \                                                 │
│        4   5                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 二叉树遍历模板

### 前序遍历（根-左-右）

```python
def preorder(root):
    if not root:
        return
    print(root.val)      # 先处理根
    preorder(root.left)  # 再左子树
    preorder(root.right) # 后右子树
```

### 中序遍历（左-根-右）

```python
def inorder(root):
    if not root:
        return
    inorder(root.left)   # 先左子树
    print(root.val)      # 再处理根
    inorder(root.right)  # 后右子树
```

### 后序遍历（左-右-根）

```python
def postorder(root):
    if not root:
        return
    postorder(root.left)  # 先左子树
    postorder(root.right) # 再右子树
    print(root.val)       # 后处理根
```

### 层序遍历（BFS）

```python
from collections import deque

def levelorder(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result
```

---

## 1️⃣ LC 104. 二叉树的最大深度 🟢

### 题目描述
返回二叉树的最大深度。

### 🎨 图解思路

```
    3
   / \
  9  20
    /  \
   15   7

分解思维:
maxDepth(3) = 1 + max(maxDepth(9), maxDepth(20))
            = 1 + max(1, 2)
            = 3
```

### 💻 代码实现

```python
def maxDepth(root) -> int:
    if not root:
        return 0
    
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    
    return 1 + max(left_depth, right_depth)
```

### 🧠 记忆口诀
> **"深度 = 1 + max(左深度, 右深度)"**

---

## 2️⃣ LC 100. 相同的树 🟢

### 题目描述
判断两棵树是否相同。

### 🎨 图解思路

```
   p       q
   1       1
  / \     / \
 2   3   2   3

相同的条件:
1. 根节点值相同
2. 左子树相同
3. 右子树相同
```

### 💻 代码实现

```python
def isSameTree(p, q) -> bool:
    if not p and not q:
        return True
    if not p or not q:
        return False
    
    return (p.val == q.val and 
            isSameTree(p.left, q.left) and 
            isSameTree(p.right, q.right))
```

### 🧠 记忆口诀
> **"根同左同右同，才是真的同"**

---

## 3️⃣ LC 226. 翻转二叉树 🟢

### 题目描述
翻转二叉树（镜像）。

### 🎨 图解思路

```
     4              4
   /   \          /   \
  2     7   =>   7     2
 / \   / \      / \   / \
1   3 6   9    9   6 3   1

交换每个节点的左右子树
```

### 💻 代码实现

```python
def invertTree(root):
    if not root:
        return None
    
    # 交换左右子树
    root.left, root.right = root.right, root.left
    
    # 递归翻转子树
    invertTree(root.left)
    invertTree(root.right)
    
    return root
```

### 🧠 记忆口诀
> **"先交换，再递归"**

---

## 4️⃣ LC 101. 对称二叉树 🟢

### 题目描述
判断二叉树是否对称。

### 🎨 图解思路

```
    1
   / \
  2   2
 / \ / \
3  4 4  3

对称条件:
左子树的左 == 右子树的右
左子树的右 == 右子树的左
```

### 💻 代码实现

```python
def isSymmetric(root) -> bool:
    def check(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        
        return (left.val == right.val and
                check(left.left, right.right) and
                check(left.right, right.left))
    
    return check(root.left, root.right) if root else True
```

### 🧠 记忆口诀
> **"外外相等，内内相等"**

---

## 5️⃣ LC 105. 从前序与中序遍历序列构造二叉树 🟡

### 题目描述
根据前序和中序遍历结果，构建二叉树。

### 🎨 图解思路

```
preorder = [3, 9, 20, 15, 7]  根-左-右
inorder  = [9, 3, 15, 20, 7]  左-根-右

步骤:
1. preorder[0] = 3 是根节点
2. 在 inorder 中找到 3，左边是左子树，右边是右子树
3. 递归构建

    3
   / \
  9  20
    /  \
   15   7
```

### 💻 代码实现

```python
def buildTree(preorder: list, inorder: list):
    if not preorder:
        return None
    
    # 根节点是前序第一个
    root = TreeNode(preorder[0])
    
    # 在中序中找到根节点位置
    mid = inorder.index(preorder[0])
    
    # 递归构建左右子树
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])
    
    return root
```

### 🧠 记忆口诀
> **"前序定根，中序分边"**

---

## 6️⃣ LC 106. 从中序与后序遍历序列构造二叉树 🟡

### 题目描述
根据中序和后序遍历结果，构建二叉树。

### 🎨 图解思路

```
inorder   = [9, 3, 15, 20, 7]  左-根-右
postorder = [9, 15, 7, 20, 3]  左-右-根

后序最后一个是根！
```

### 💻 代码实现

```python
def buildTree(inorder: list, postorder: list):
    if not postorder:
        return None
    
    # 根节点是后序最后一个
    root = TreeNode(postorder[-1])
    
    # 在中序中找到根节点位置
    mid = inorder.index(postorder[-1])
    
    # 递归构建左右子树
    root.left = buildTree(inorder[:mid], postorder[:mid])
    root.right = buildTree(inorder[mid+1:], postorder[mid:-1])
    
    return root
```

### 🧠 记忆口诀
> **"后序定根（最后），中序分边"**

---

## 7️⃣ LC 117. 填充每个节点的下一个右侧节点指针 II 🟡

### 题目描述
填充每个节点的 next 指针指向右侧节点。

### 🎨 图解思路

```
     1 → NULL
   /   \
  2  →  3 → NULL
 / \     \
4→  5  →  7 → NULL

使用层序遍历，连接同层节点
```

### 💻 代码实现

```python
def connect(root):
    if not root:
        return None
    
    queue = deque([root])
    
    while queue:
        size = len(queue)
        prev = None
        
        for i in range(size):
            node = queue.popleft()
            
            if prev:
                prev.next = node
            prev = node
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return root
```

### 🧠 记忆口诀
> **"层序遍历，前连后"**

---

## 8️⃣ LC 114. 二叉树展开为链表 🟡

### 题目描述
将二叉树展开为单链表（前序顺序）。

### 🎨 图解思路

```
    1           1
   / \           \
  2   5    =>     2
 / \   \           \
3   4   6           3
                     \
                      4
                       \
                        5
                         \
                          6
```

### 💻 代码实现

```python
def flatten(root) -> None:
    if not root:
        return
    
    # 后序遍历：先处理子树，再处理根
    flatten(root.left)
    flatten(root.right)
    
    # 保存右子树
    right = root.right
    
    # 左子树移到右边
    root.right = root.left
    root.left = None
    
    # 找到右子树末端，接上原右子树
    while root.right:
        root = root.right
    root.right = right
```

### 🧠 记忆口诀
> **"左接右，原右接末尾"**

---

## 9️⃣ LC 112. 路径总和 🟢

### 题目描述
判断是否存在根到叶子路径，其和等于目标值。

### 🎨 图解思路

```
      5
     / \
    4   8
   /   / \
  11  13  4
 /  \      \
7    2      1

targetSum = 22
路径: 5 → 4 → 11 → 2 = 22 ✓
```

### 💻 代码实现

```python
def hasPathSum(root, targetSum: int) -> bool:
    if not root:
        return False
    
    # 叶子节点
    if not root.left and not root.right:
        return root.val == targetSum
    
    # 递归检查子树
    remaining = targetSum - root.val
    return (hasPathSum(root.left, remaining) or 
            hasPathSum(root.right, remaining))
```

### 🧠 记忆口诀
> **"叶子判相等，非叶递归减"**

---

## 🔟 LC 129. 求根节点到叶节点数字之和 🟡

### 题目描述
每条路径组成一个数字，求所有数字之和。

### 🎨 图解思路

```
    1
   / \
  2   3

路径: 1→2 = 12
路径: 1→3 = 13
总和: 12 + 13 = 25
```

### 💻 代码实现

```python
def sumNumbers(root) -> int:
    def dfs(node, current_sum):
        if not node:
            return 0
        
        current_sum = current_sum * 10 + node.val
        
        # 叶子节点
        if not node.left and not node.right:
            return current_sum
        
        return dfs(node.left, current_sum) + dfs(node.right, current_sum)
    
    return dfs(root, 0)
```

### 🧠 记忆口诀
> **"进位乘10加当前"**

---

## 1️⃣1️⃣ LC 124. 二叉树中的最大路径和 🔴

### 题目描述
找出路径和最大的路径（可以不经过根节点）。

### 🎨 图解思路

```
   -10
   /  \
  9   20
     /  \
    15   7

最大路径: 15 → 20 → 7 = 42

思路:
每个节点可以：
1. 只贡献自己（作为路径端点）
2. 贡献自己+左子树
3. 贡献自己+右子树
4. 作为拐点（左+自己+右）
```

### 💻 代码实现

```python
def maxPathSum(root) -> int:
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # 左右子树的最大贡献（负数不要）
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # 当前节点作为拐点的路径和
        path_sum = node.val + left_gain + right_gain
        max_sum = max(max_sum, path_sum)
        
        # 返回给父节点的贡献（只能选一边）
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum
```

### 🧠 记忆口诀
> **"拐点算全局，贡献选一边"**

---

## 1️⃣2️⃣ LC 173. 二叉搜索树迭代器 🟡

### 题目描述
实现二叉搜索树的迭代器。

### 💻 代码实现

```python
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self._leftmost_inorder(root)
    
    def _leftmost_inorder(self, node):
        while node:
            self.stack.append(node)
            node = node.left
    
    def next(self) -> int:
        node = self.stack.pop()
        if node.right:
            self._leftmost_inorder(node.right)
        return node.val
    
    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

### 🧠 记忆口诀
> **"栈存左链，弹出处理右"**

---

## 1️⃣3️⃣ LC 222. 完全二叉树的节点个数 🟢

### 题目描述
统计完全二叉树的节点个数。

### 💻 代码实现

```python
def countNodes(root) -> int:
    if not root:
        return 0
    
    left_depth = right_depth = 0
    left, right = root, root
    
    while left:
        left_depth += 1
        left = left.left
    
    while right:
        right_depth += 1
        right = right.right
    
    # 满二叉树
    if left_depth == right_depth:
        return 2 ** left_depth - 1
    
    # 递归
    return 1 + countNodes(root.left) + countNodes(root.right)
```

### 🧠 记忆口诀
> **"满树用公式，不满递归数"**

---

## 1️⃣4️⃣ LC 236. 二叉树的最近公共祖先 🟡

### 题目描述
找两个节点的最近公共祖先（LCA）。

### 🎨 图解思路

```
        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4

LCA(5, 1) = 3
LCA(5, 4) = 5
```

### 💻 代码实现

```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    # p, q 分布在两边
    if left and right:
        return root
    
    # p, q 在同一边
    return left if left else right
```

### 🧠 记忆口诀
> **"左右都有返回根，否则返回有的那边"**

---

## 📊 本章总结

### 题目速查表

| 题号 | 题目 | 难度 | 类型 |
|------|------|------|------|
| 104 | 最大深度 | 🟢 | 深度 |
| 100 | 相同的树 | 🟢 | 比较 |
| 226 | 翻转二叉树 | 🟢 | 变换 |
| 101 | 对称二叉树 | 🟢 | 比较 |
| 105 | 前序+中序构造 | 🟡 | 构造 |
| 106 | 中序+后序构造 | 🟡 | 构造 |
| 117 | 填充next指针 | 🟡 | 层序 |
| 114 | 展开为链表 | 🟡 | 变换 |
| 112 | 路径总和 | 🟢 | 路径 |
| 129 | 数字之和 | 🟡 | 路径 |
| 124 | 最大路径和 | 🔴 | 路径 |
| 173 | BST迭代器 | 🟡 | 迭代 |
| 222 | 完全树节点数 | 🟢 | 计数 |
| 236 | 最近公共祖先 | 🟡 | LCA |

### 🧠 全章记忆口诀

```
深度相同翻对称
前中后序建树型
连接展开走路径
迭代计数找祖宗

深度 - 最大深度 (104)
相同 - 相同的树 (100)
翻 - 翻转二叉树 (226)
对称 - 对称二叉树 (101)
前中后序 - 从遍历构造 (105, 106)
连接 - 填充next指针 (117)
展开 - 展开为链表 (114)
路径 - 路径总和系列 (112, 129, 124)
迭代 - BST迭代器 (173)
计数 - 节点个数 (222)
祖宗 - 最近公共祖先 (236)
```

---

> 📖 **下一篇**：[二叉搜索树专题](/2026/01/18/leetcode-150-bst/)

