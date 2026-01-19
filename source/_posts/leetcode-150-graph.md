---
title: 🔍 LeetCode 150 - 图论与搜索专题
date: 2026-01-18
updated: 2026-01-18
categories: 
  - 算法
  - LeetCode
tags: 
  - LeetCode
  - leetcode面试150
  - 图论
  - BFS
  - DFS
  - 拓扑排序
  - 面试
description: LeetCode 面试 150 题之图论与搜索专题，含BFS/DFS图解、代码模板、记忆口诀
---

# 🔍 图论与搜索专题 (8题)

> 🎯 **核心技巧**：BFS、DFS、拓扑排序、并查集

---

## 🗺️ 图搜索算法对比

```
┌─────────────────────────────────────────────────────────────┐
│                    BFS vs DFS                                │
├──────────────────────┬──────────────────────────────────────┤
│         BFS          │              DFS                      │
│    (广度优先搜索)     │         (深度优先搜索)                │
├──────────────────────┼──────────────────────────────────────┤
│  数据结构：队列       │  数据结构：栈/递归                    │
│  探索方式：层层扩展   │  探索方式：一路走到底                 │
│  适用：最短路径       │  适用：连通性、路径搜索               │
├──────────────────────┼──────────────────────────────────────┤
│       1              │         1                            │
│      /|\             │        /|\                           │
│     2 3 4  → 层序    │       2 3 4  → 深入                  │
│    /|   |            │      /|   |                          │
│   5 6   7            │     5 6   7                          │
│                      │                                       │
│  顺序: 1→2→3→4→5→6→7 │  顺序: 1→2→5→6→3→4→7                 │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 🔧 BFS 模板

```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = {start}
    level = 0
    
    while queue:
        size = len(queue)
        for _ in range(size):
            node = queue.popleft()
            
            # 处理当前节点
            process(node)
            
            # 将邻居加入队列
            for neighbor in get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        level += 1  # 层数 +1
    
    return level
```

## 🔧 DFS 模板

```python
def dfs(node, visited):
    if node in visited:
        return
    
    visited.add(node)
    
    # 处理当前节点
    process(node)
    
    # 递归访问邻居
    for neighbor in get_neighbors(node):
        dfs(neighbor, visited)
```

---

## 1️⃣ LC 200. 岛屿数量 🟡

### 题目描述
计算二维网格中岛屿的数量（由 '1' 组成的连通区域）。

### 🎨 图解思路

```
grid:
1 1 0 0 0
1 1 0 0 0
0 0 1 0 0
0 0 0 1 1

岛屿数量 = 3

策略：遍历网格，遇到 '1' 就启动 DFS/BFS
把整个岛屿标记为已访问，计数 +1
```

### 💻 代码实现 (DFS)

```python
def numIslands(grid: list) -> int:
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    count = 0
    
    def dfs(i, j):
        # 边界检查 & 是否为陆地
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        
        # 标记为已访问
        grid[i][j] = '0'
        
        # 四个方向扩展
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    
    return count
```

### 🧠 记忆口诀
> **"遇1就淹，淹完计数"**

---

## 2️⃣ LC 130. 被围绕的区域 🟡

### 题目描述
将所有被 'X' 围绕的 'O' 填充为 'X'（边界上的 'O' 及其连通的 'O' 不算被围绕）。

### 🎨 图解思路

```
输入:               输出:
X X X X            X X X X
X O O X      →     X X X X
X X O X            X X X X
X O X X            X O X X

逆向思维：
1. 从边界的 'O' 开始 DFS，标记为 '#'
2. 遍历整个网格：
   - 'O' → 'X' (被围绕)
   - '#' → 'O' (恢复)
```

### 💻 代码实现

```python
def solve(board: list) -> None:
    if not board:
        return
    
    m, n = len(board), len(board[0])
    
    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != 'O':
            return
        board[i][j] = '#'
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)
    
    # 从边界开始标记
    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for j in range(n):
        dfs(0, j)
        dfs(m - 1, j)
    
    # 恢复和填充
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                board[i][j] = 'X'
            elif board[i][j] == '#':
                board[i][j] = 'O'
```

### 🧠 记忆口诀
> **"边界O不围，标记后恢复"**

---

## 3️⃣ LC 133. 克隆图 🟡

### 题目描述
深拷贝一个无向连通图。

### 🎨 图解思路

```
原图:          克隆:
  1 --- 2       1' --- 2'
  |     |       |      |
  4 --- 3       4' --- 3'

使用哈希表记录 原节点 → 克隆节点 的映射
BFS 或 DFS 遍历并克隆
```

### 💻 代码实现

```python
def cloneGraph(node: 'Node') -> 'Node':
    if not node:
        return None
    
    # 哈希表：原节点 → 克隆节点
    cloned = {}
    
    def dfs(node):
        if node in cloned:
            return cloned[node]
        
        # 创建克隆节点
        clone = Node(node.val)
        cloned[node] = clone
        
        # 克隆邻居
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

### 🧠 记忆口诀
> **"哈希记映射，DFS克隆"**

---

## 4️⃣ LC 399. 除法求值 🟡

### 题目描述
给定变量对的除法结果，求解其他除法。

### 🎨 图解思路

```
equations: [["a","b"],["b","c"]]
values: [2.0, 3.0]

构建带权图:
a --2.0--> b --3.0--> c
a <--0.5-- b <--0.33-- c

查询 a/c = a/b * b/c = 2.0 * 3.0 = 6.0
使用 BFS 找路径并累乘权重
```

### 💻 代码实现

```python
def calcEquation(equations, values, queries):
    from collections import defaultdict, deque
    
    # 构建图
    graph = defaultdict(dict)
    for (a, b), val in zip(equations, values):
        graph[a][b] = val
        graph[b][a] = 1.0 / val
    
    def bfs(start, end):
        if start not in graph or end not in graph:
            return -1.0
        if start == end:
            return 1.0
        
        queue = deque([(start, 1.0)])
        visited = {start}
        
        while queue:
            node, product = queue.popleft()
            
            for neighbor, weight in graph[node].items():
                if neighbor == end:
                    return product * weight
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, product * weight))
        
        return -1.0
    
    return [bfs(a, b) for a, b in queries]
```

### 🧠 记忆口诀
> **"带权图建边，BFS累乘"**

---

## 5️⃣ LC 207. 课程表 🟡

### 题目描述
判断是否可能完成所有课程（检测有向图是否有环）。

### 🎨 图解思路

```
numCourses = 4
prerequisites = [[1,0],[2,0],[3,1],[3,2]]

构建图:
0 → 1 → 3
0 → 2 ↗

拓扑排序：
1. 统计每个节点的入度
2. 将入度为 0 的节点入队
3. BFS 处理，每处理一个节点，邻居入度 -1
4. 入度变为 0 的节点入队
5. 如果处理的节点数 = 总课程数，则无环
```

### 💻 代码实现

```python
def canFinish(numCourses: int, prerequisites: list) -> bool:
    from collections import defaultdict, deque
    
    # 构建图和入度表
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # 将入度为 0 的节点入队
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0
    
    while queue:
        node = queue.popleft()
        count += 1
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == numCourses
```

### 🧠 记忆口诀
> **"入度为零先处理，全部处理完无环"**

---

## 6️⃣ LC 210. 课程表 II 🟡

### 题目描述
返回完成所有课程的学习顺序（拓扑排序结果）。

### 💻 代码实现

```python
def findOrder(numCourses: int, prerequisites: list) -> list:
    from collections import defaultdict, deque
    
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == numCourses else []
```

### 🧠 记忆口诀
> **"拓扑排序，记录顺序"**

---

## 7️⃣ LC 909. 蛇梯棋 🟡

### 题目描述
在蛇梯棋盘上，求从起点到终点的最少移动次数。

### 🎨 图解思路

```
棋盘按 Boustrophedon 方式编号（蛇形）

使用 BFS 求最短路径
每次可以走 1-6 步（掷骰子）
遇到蛇/梯子则传送到对应位置
```

### 💻 代码实现

```python
def snakesAndLadders(board: list) -> int:
    from collections import deque
    
    n = len(board)
    
    # 将位置编号转为坐标
    def get_position(num):
        num -= 1
        row = n - 1 - num // n
        col = num % n if (n - 1 - row) % 2 == 0 else n - 1 - num % n
        return row, col
    
    queue = deque([(1, 0)])  # (位置, 步数)
    visited = {1}
    
    while queue:
        curr, steps = queue.popleft()
        
        for dice in range(1, 7):
            next_pos = curr + dice
            
            if next_pos > n * n:
                continue
            
            r, c = get_position(next_pos)
            if board[r][c] != -1:
                next_pos = board[r][c]
            
            if next_pos == n * n:
                return steps + 1
            
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, steps + 1))
    
    return -1
```

### 🧠 记忆口诀
> **"BFS找最短，蛇梯要传送"**

---

## 8️⃣ LC 433. 最小基因变化 🟡

### 题目描述
从起始基因变化到目标基因的最少变化次数。

### 🎨 图解思路

```
startGene = "AACCGGTT"
endGene = "AAACGGTA"
bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]

每次只能变一个字符，且结果必须在 bank 中
使用 BFS 搜索最短路径
```

### 💻 代码实现

```python
def minMutation(startGene: str, endGene: str, bank: list) -> int:
    from collections import deque
    
    bank_set = set(bank)
    if endGene not in bank_set:
        return -1
    
    queue = deque([(startGene, 0)])
    visited = {startGene}
    chars = ['A', 'C', 'G', 'T']
    
    while queue:
        gene, steps = queue.popleft()
        
        if gene == endGene:
            return steps
        
        for i in range(len(gene)):
            for c in chars:
                if c != gene[i]:
                    new_gene = gene[:i] + c + gene[i+1:]
                    
                    if new_gene in bank_set and new_gene not in visited:
                        visited.add(new_gene)
                        queue.append((new_gene, steps + 1))
    
    return -1
```

### 🧠 记忆口诀
> **"每次变一个，在库中才行"**

---

## 📊 本章总结

### 图搜索场景选择

| 场景 | 推荐算法 | 典型题目 |
|------|----------|----------|
| 最短路径 | BFS | 909, 433 |
| 连通性 | DFS/BFS | 200, 130 |
| 拓扑排序 | BFS (Kahn) | 207, 210 |
| 图克隆 | DFS + 哈希 | 133 |
| 带权路径 | BFS + 累乘 | 399 |

### 🧠 全章记忆口诀

```
岛围克除课课蛇基
图论八题记仔细

岛 - 岛屿数量 (200)
围 - 被围绕的区域 (130)
克 - 克隆图 (133)
除 - 除法求值 (399)
课课 - 课程表 I/II (207, 210)
蛇 - 蛇梯棋 (909)
基 - 最小基因变化 (433)
```

---

> 📖 **返回**：[LeetCode 150 题总目录](/2026/01/18/leetcode-150-index/)

