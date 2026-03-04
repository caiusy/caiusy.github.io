---
title: 图论与搜索：从零到精通的费曼式完全指南
date: 2026-02-23 06:38:34
tags:
  - 算法
  - 图论
  - DFS
  - BFS
  - LeetCode
categories:
  - 算法与数据结构
mathjax: true
---

> **费曼说：** "如果你不能把一个概念解释给一个 10 岁小孩听，说明你自己也没真正理解它。"

这篇文章用费曼式教学法，带你从「图是什么」一路打通到「LeetCode 实战」。每个概念都有具体例子、可视化图解、完整代码和数据流追踪。不讲废话，直击本质。

<!-- more -->

---

## 第一章：基础篇 — 图的世界观

---

### 1.1 图到底是什么？

**费曼一句话：** 图就是「一堆东西」加上「它们之间的关系」。就像你的朋友圈——人是节点，互相认识就连条线。

形式化定义：图 $G = (V, E)$，其中 $V$ 是节点集合，$E$ 是边集合。

但定义不重要，重要的是：**你怎么在电脑里存一张图？**

这就引出了三种表示法。

---

### 1.2 图的三种表示法

我们用这张图作为贯穿全文的例子：

![示例图](/images/graph-theory/02_example_graph.png)

5 个节点（0-4），6 条边。下面分别看三种存法。

![三种表示法对比](/images/graph-theory/01_three_representations.png)

#### 1.2.1 邻接矩阵 (Adjacency Matrix)

**费曼一句话：** 画一张表格，行和列都是节点。两个节点之间有边就填 1，没有就填 0。就像一张「谁认识谁」的关系表。

```python
# 邻接矩阵表示
V = 5
adj_matrix = [[0]*V for _ in range(V)]
edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]

for u, v in edges:
    adj_matrix[u][v] = 1
    adj_matrix[v][u] = 1  # 无向图，对称

# 查询: 节点1和节点3之间有边吗？
print(adj_matrix[1][3])  # 输出: 1, O(1)时间
```

**数据流追踪：**
```
初始矩阵: 5x5 全0
加边(0,1): matrix[0][1]=1, matrix[1][0]=1
加边(0,2): matrix[0][2]=1, matrix[2][0]=1
...
最终矩阵:
    0  1  2  3  4
0 [ 0, 1, 1, 0, 0 ]
1 [ 1, 0, 1, 1, 0 ]
2 [ 1, 1, 0, 0, 1 ]
3 [ 0, 1, 0, 0, 1 ]
4 [ 0, 0, 1, 1, 0 ]
```

**什么时候用？**
- ✅ 需要快速查询「两点之间有没有边」→ O(1)
- ✅ 稠密图（边很多，接近 $V^2$）
- ❌ 稀疏图浪费空间 → O(V²) 空间，大部分是 0

#### 1.2.2 邻接表 (Adjacency List)

**费曼一句话：** 每个人维护一个「好友列表」。想知道某人认识谁，直接翻他的列表就行。

```python
# 邻接表表示 (最常用!)
from collections import defaultdict

graph = defaultdict(list)
edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]

for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# 查询: 节点1的所有邻居？
print(graph[1])  # 输出: [0, 2, 3], O(degree)时间
```

**数据流追踪：**
```
加边(0,1): graph = {0:[1], 1:[0]}
加边(0,2): graph = {0:[1,2], 1:[0], 2:[0]}
加边(1,2): graph = {0:[1,2], 1:[0,2], 2:[0,1]}
加边(1,3): graph = {0:[1,2], 1:[0,2,3], 2:[0,1], 3:[1]}
加边(2,4): graph = {0:[1,2], 1:[0,2,3], 2:[0,1,4], 3:[1], 4:[2]}
加边(3,4): graph = {0:[1,2], 1:[0,2,3], 2:[0,1,4], 3:[1,4], 4:[2,3]}
```

**什么时候用？**
- ✅ 遍历邻居快 → O(degree)
- ✅ 空间高效 → O(V+E)
- ✅ **90% 的图题都用这个**
- ❌ 查询特定边是否存在 → O(degree)，不如矩阵快

#### 1.2.3 边列表 (Edge List)

**费曼一句话：** 最笨但最直接——把所有关系一条条列出来。就像一份「谁和谁是朋友」的名单。

```python
# 边列表表示
edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]

# 带权重的边列表 (Kruskal最小生成树常用)
weighted_edges = [(0,1,4),(0,2,2),(1,2,1),(1,3,5),(2,4,3),(3,4,6)]
weighted_edges.sort(key=lambda x: x[2])  # 按权重排序
```

**什么时候用？**
- ✅ Kruskal 最小生成树（需要按权重排序所有边）
- ✅ 存储简单，空间 O(E)
- ❌ 查边、查邻居都慢 → O(E)

#### 三种表示法对比总结

| 特性 | 邻接矩阵 | 邻接表 | 边列表 |
|------|----------|--------|--------|
| 空间 | O(V²) | O(V+E) | O(E) |
| 查边 | **O(1)** | O(degree) | O(E) |
| 遍历邻居 | O(V) | **O(degree)** | O(E) |
| 加边 | O(1) | O(1) | O(1) |
| 适用场景 | 稠密图 | **通用(首选)** | Kruskal |

> 🎯 **结论：刷题默认用邻接表。** 除非题目明确给了矩阵（如岛屿问题的网格），或者需要 Kruskal。

---

### 1.3 DFS 深度优先搜索

**费曼一句话：** DFS 就是走迷宫的策略——一条路走到黑，撞墙了再回头换一条。用「栈」记住岔路口，方便回头。

#### 核心机制

DFS 的本质是：**优先探索深度方向**。它用栈（递归调用栈或显式栈）来记录「还没探索完的岔路」。

关键数据结构：
- `stack`：待探索的节点（后进先出）
- `visited`：已经去过的节点（避免重复走）

#### 递归写法（利用系统调用栈）

```python
def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(f"访问: {node}")
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
    return visited
```

**数据流追踪（递归版）：**
```
调用 dfs(graph, 0)
  visited={0}, 打印"访问: 0"
  邻居: [1, 2]
  ├─ 1 未访问 → 调用 dfs(graph, 1)
  │    visited={0,1}, 打印"访问: 1"
  │    邻居: [0, 2, 3]
  │    ├─ 0 已访问 → 跳过
  │    ├─ 2 未访问 → 调用 dfs(graph, 2)
  │    │    visited={0,1,2}, 打印"访问: 2"
  │    │    邻居: [0, 1, 4]
  │    │    ├─ 0 已访问 → 跳过
  │    │    ├─ 1 已访问 → 跳过
  │    │    └─ 4 未访问 → 调用 dfs(graph, 4)
  │    │         visited={0,1,2,4}, 打印"访问: 4"
  │    │         邻居: [2, 3]
  │    │         ├─ 2 已访问 → 跳过
  │    │         └─ 3 未访问 → 调用 dfs(graph, 3)
  │    │              visited={0,1,2,3,4}, 打印"访问: 3"
  │    │              邻居: [1, 4] → 全部已访问
  │    │              返回 ↩
  │    │         返回 ↩
  │    │    返回 ↩
  │    └─ 3 已访问 → 跳过
  │    返回 ↩
  └─ 2 已访问 → 跳过
  返回 ↩

最终遍历顺序: 0 → 1 → 2 → 4 → 3
```

#### 迭代写法（显式栈）

```python
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    order = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        # 逆序压栈，保证小编号先被弹出
        for neighbor in reversed(sorted(graph[node])):
            if neighbor not in visited:
                stack.append(neighbor)
    return order
```

#### 栈的变化过程（逐步图解）

![DFS 逐步过程](/images/graph-theory/03_dfs_steps.png)

![栈 vs 队列变化](/images/graph-theory/06_stack_vs_queue.png)

**迭代版数据流追踪：**
```
初始: stack=[0], visited={}

Step1: pop 0 → visited={0}, 压入邻居[2,1]
       stack=[2,1]

Step2: pop 1 → visited={0,1}, 压入邻居[3,2]
       stack=[2,3,2]

Step3: pop 2 → visited={0,1,2}, 压入邻居[4]
       stack=[2,3,4]

Step4: pop 4 → visited={0,1,2,4}, 压入邻居[3]
       stack=[2,3,3]

Step5: pop 3 → visited={0,1,2,3,4}
       stack=[2,3]

Step6: pop 3 → 已访问,跳过. pop 2 → 已访问,跳过
       stack=[] → 结束

遍历顺序: 0 → 1 → 2 → 4 → 3
```

> ⚠️ **递归 vs 迭代的区别：** 递归版遍历顺序取决于邻居的遍历顺序；迭代版因为栈的 LIFO 特性，需要逆序压栈才能保持相同顺序。两者本质相同，只是栈的管理方式不同。

#### DFS 的时间/空间复杂度

- 时间：O(V + E) — 每个节点访问一次，每条边检查一次
- 空间：O(V) — visited 集合 + 栈深度（最坏情况是链状图，深度为 V）

---

### 1.4 BFS 广度优先搜索

**费曼一句话：** BFS 就像往池塘里扔石头——水波一圈一圈往外扩。先把离你最近的人全认识了，再去认识朋友的朋友。用「队列」保证先来先服务。

#### 核心机制

BFS 的本质是：**按距离（层数）从近到远探索**。它用队列（先进先出）来保证「先发现的节点先处理」。

关键数据结构：
- `queue`：待探索的节点（先进先出）
- `visited`：已经入过队的节点（注意：是入队时标记，不是出队时！）

#### 标准写法

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)  # 入队时就标记!
                queue.append(neighbor)
    return order
```

> ⚠️ **关键细节：visited 在入队时标记，不是出队时！** 如果出队时才标记，同一个节点可能被多次入队，浪费时间和空间。这是新手最常犯的错误。

#### 队列的变化过程（逐步图解）

![BFS 逐步过程](/images/graph-theory/04_bfs_steps.png)

**数据流追踪：**
```
初始: queue=[0], visited={0}

Step1: popleft 0 → 处理邻居 [1,2]
       1 未访问 → visited={0,1}, 入队
       2 未访问 → visited={0,1,2}, 入队
       queue=[1,2]                          ← Layer 0 处理完

Step2: popleft 1 → 处理邻居 [0,2,3]
       0 已访问 → 跳过
       2 已访问 → 跳过
       3 未访问 → visited={0,1,2,3}, 入队
       queue=[2,3]

Step3: popleft 2 → 处理邻居 [0,1,4]
       0,1 已访问 → 跳过
       4 未访问 → visited={0,1,2,3,4}, 入队
       queue=[3,4]                          ← Layer 1 处理完

Step4: popleft 3 → 邻居 [1,4] 全已访问
       queue=[4]

Step5: popleft 4 → 邻居 [2,3] 全已访问
       queue=[] → 结束                      ← Layer 2 处理完

遍历顺序: 0 → 1 → 2 → 3 → 4
层次:     L0    L1  L1   L2  L2
```

#### 层序遍历的本质

BFS 天然按层遍历。如果你需要知道「当前是第几层」，只需要在每层开始时记录队列长度：

```python
from collections import deque

def bfs_by_layer(graph, start):
    visited = {start}
    queue = deque([start])
    depth = 0
    while queue:
        layer_size = len(queue)  # 当前层有多少节点
        print(f"Layer {depth}: ", end="")
        for _ in range(layer_size):
            node = queue.popleft()
            print(node, end=" ")
            for nb in sorted(graph[node]):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        print()
        depth += 1
```

```
输出:
Layer 0: 0
Layer 1: 1 2
Layer 2: 3 4
```

> 🎯 **为什么 BFS 能求最短路？** 因为 BFS 按层扩展，第一次到达某个节点时，走的一定是最少的边数。这就是无权图最短路的原理。

#### BFS 的时间/空间复杂度

- 时间：O(V + E) — 和 DFS 一样
- 空间：O(V) — 最坏情况队列里存了一整层的节点（完全图时接近 V）

---

### 1.5 DFS vs BFS 全面对比

![DFS vs BFS 遍历树](/images/graph-theory/05_dfs_vs_bfs_tree.png)

| 维度 | DFS | BFS |
|------|-----|-----|
| 数据结构 | 栈 (LIFO) | 队列 (FIFO) |
| 探索策略 | 一条路走到底 | 一层一层扩展 |
| 空间复杂度 | O(h), h=最大深度 | O(w), w=最大宽度 |
| 能求最短路？ | ❌ 不能 | ✅ 无权图最短路 |
| 适合场景 | 连通性、回溯、拓扑排序 | 最短路、层序遍历 |
| 类比 | 走迷宫（一条路走到黑） | 水波扩散（一圈圈往外） |

> 🎯 **选择口诀：** 要找最短路 → BFS。要遍历所有路径/判断连通性 → DFS。不确定 → 两个都能用，选你顺手的。

---

### 1.6 新手常踩的 5 个坑

**坑1: BFS 的 visited 标记时机**

```python
# ❌ 错误: 出队时标记
while queue:
    node = queue.popleft()
    if node in visited: continue  # 太晚了! 可能已经入队多次
    visited.add(node)

# ✅ 正确: 入队时标记
if neighbor not in visited:
    visited.add(neighbor)  # 入队前就标记
    queue.append(neighbor)
```

出队时标记会导致同一节点被多次入队。想象 A 和 B 都连着 C，处理 A 时把 C 入队，处理 B 时又把 C 入队——C 被处理两次。入队时标记就能避免。

**坑2: DFS 递归爆栈**

Python 默认递归深度 1000。如果图是一条链（比如 10000 个节点串成一条线），递归 DFS 直接 `RecursionError`。

解决方案：
```python
import sys
sys.setrecursionlimit(200000)  # 方案1: 加大限制(不推荐)

# 方案2: 改用迭代DFS (推荐)
def dfs_iterative(graph, start):
    stack, visited = [start], set()
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        for nb in graph[node]:
            if nb not in visited:
                stack.append(nb)
```

**坑3: 有向图 vs 无向图建图**

```python
# 无向图: 双向加边
graph[u].append(v)
graph[v].append(u)

# 有向图: 单向加边
graph[u].append(v)  # 只有 u→v
```

题目说「无向」就必须双向加，漏了一个方向会导致遍历不完整。

**坑4: 网格图的边界检查**

```python
# 模板: 四方向遍历
for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
    nx, ny = x+dx, y+dy
    if 0 <= nx < m and 0 <= ny < n:  # 边界检查
        # 处理 (nx, ny)
```

忘记边界检查 → IndexError。建议把方向数组写成常量，减少手误。

**坑5: Dijkstra 忘记跳过过时条目**

```python
d, u = heapq.heappop(heap)
if d > dist.get(u, float('inf')):
    continue  # 这个条目已经过时了，跳过!
```

堆里可能有同一节点的多个条目（因为松弛时 push 新的而不是更新旧的）。不跳过会导致重复处理，虽然结果正确但时间退化。

---

### 1.7 DFS/BFS 模板速查表

**DFS 万能模板（迭代版）：**
```python
def dfs(graph, start):
    stack, visited = [start], {start}
    while stack:
        node = stack.pop()
        # process(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
```

**BFS 万能模板（带层数）：**
```python
def bfs(graph, start):
    queue, visited = deque([start]), {start}
    depth = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            # process(node, depth)
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        depth += 1
```

**网格 DFS 模板：**
```python
def grid_dfs(grid, i, j, m, n):
    if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != target:
        return
    grid[i][j] = visited_mark  # 标记
    for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
        grid_dfs(grid, i+di, j+dj, m, n)
```

---

## 📝 基础篇小结

到这里你已经掌握了：
1. **图的三种存法** — 邻接表是默认选择
2. **DFS** — 栈驱动，一条路走到黑，递归和迭代两种写法
3. **BFS** — 队列驱动，一层层扩展，天然求最短路

这三个是图论的地基。接下来的技巧篇（拓扑排序、连通分量、二分图、最短路）全部建立在 DFS/BFS 之上。

---

## 第二章：技巧篇 — 图的经典应用

---

### 2.1 拓扑排序

**费曼一句话：** 你有一堆课要上，有些课有先修要求。拓扑排序就是帮你排出一个「不违反任何先修要求」的选课顺序。如果排不出来，说明先修关系里有死循环（环）。

#### 前提条件

拓扑排序只对 **有向无环图 (DAG)** 有意义。如果图里有环，就不可能排出合法顺序。

我们用这个课程依赖图：
```
0 → 1 → 3 → 4
0 → 2 → 3
    1 → 4
```
含义：上课程 3 之前，必须先上 1 和 2；上 1 之前必须先上 0。

#### 方法一：Kahn's BFS（入度法）

**核心思想：** 不断找「没有先修要求的课」（入度为 0），上完它，然后把它从依赖关系中删掉。重复直到所有课上完。

![Kahn's BFS 拓扑排序](/images/graph-theory/07_kahn_topo.png)

```python
from collections import deque, defaultdict

def topo_sort_kahn(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue = deque(i for i in range(n) if indegree[i] == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for nb in graph[node]:
            indegree[nb] -= 1
            if indegree[nb] == 0:
                queue.append(nb)
    return result if len(result) == n else []  # 空=有环
```

**数据流追踪：**
```
初始入度: [0:0, 1:1, 2:1, 3:2, 4:2]
queue=[0], result=[]

Step1: pop 0 → result=[0]
  删边 0→1: indegree[1]=0 → 入队
  删边 0→2: indegree[2]=0 → 入队
  queue=[1,2]

Step2: pop 1 → result=[0,1]
  删边 1→3: indegree[3]=1
  删边 1→4: indegree[4]=1
  queue=[2]

Step3: pop 2 → result=[0,1,2]
  删边 2→3: indegree[3]=0 → 入队
  queue=[3]

Step4: pop 3 → result=[0,1,2,3]
  删边 3→4: indegree[4]=0 → 入队
  queue=[4]

Step5: pop 4 → result=[0,1,2,3,4]
  queue=[] → 结束

len(result)=5=n → 无环, 合法拓扑序: [0,1,2,3,4]
```

#### 方法二：DFS 后序反转

**核心思想：** DFS 递归到底再回来时，「最后完成的节点」一定是依赖链的起点。把 DFS 的完成顺序反转，就是拓扑序。

```python
def topo_sort_dfs(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    order = []
    has_cycle = False

    def dfs(u):
        nonlocal has_cycle
        color[u] = GRAY  # 正在处理
        for v in graph[u]:
            if color[v] == GRAY:  # 遇到灰色=环!
                has_cycle = True; return
            if color[v] == WHITE:
                dfs(v)
        color[u] = BLACK  # 处理完毕
        order.append(u)   # 后序: 完成时记录

    for i in range(n):
        if color[i] == WHITE:
            dfs(i)
    return order[::-1] if not has_cycle else []
```

**数据流追踪（三色标记法）：**
```
WHITE=未访问  GRAY=正在递归中  BLACK=已完成

dfs(0): color[0]=GRAY
  → dfs(1): color[1]=GRAY
    → dfs(3): color[3]=GRAY
      → dfs(4): color[4]=GRAY
        邻居都没有 → color[4]=BLACK, order=[4]
      color[3]=BLACK, order=[4,3]
    → dfs(4): 已BLACK,跳过
    color[1]=BLACK, order=[4,3,1]
  → dfs(2): color[2]=GRAY
    → dfs(3): 已BLACK,跳过
    color[2]=BLACK, order=[4,3,1,2]
  color[0]=BLACK, order=[4,3,1,2,0]

反转 → [0,2,1,3,4] ← 合法拓扑序!
```

> 🎯 **两种方法怎么选？** Kahn's BFS 更直观，能直接检测环（result 长度不够就是有环）。DFS 后序法代码更短，适合需要同时做其他事（如求强连通分量）的场景。**刷题推荐 Kahn's BFS。**

---

### 2.2 连通分量

**费曼一句话：** 一张图里可能有好几个「朋友圈」，互相之间完全不认识。每个朋友圈就是一个连通分量。找连通分量就是数「有几个独立的圈子」。

![连通分量](/images/graph-theory/08_connected_components.png)

#### 方法一：DFS 染色

从每个未访问的节点出发做一次 DFS，能到达的所有节点就是同一个连通分量。

```python
def count_components(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    count = 0

    def dfs(node):
        visited[node] = True
        for nb in graph[node]:
            if not visited[nb]:
                dfs(nb)

    for i in range(n):
        if not visited[i]:
            dfs(i)       # 发现一个新的连通分量
            count += 1
    return count
```

**数据流追踪（图: 0-1-2, 3-4, 5-6-7）：**
```
i=0: 未访问 → dfs(0)→dfs(1)→dfs(2), count=1
i=1: 已访问, 跳过
i=2: 已访问, 跳过
i=3: 未访问 → dfs(3)→dfs(4), count=2
i=4: 已访问, 跳过
i=5: 未访问 → dfs(5)→dfs(6)→dfs(7), count=3
结果: 3个连通分量
```

#### 方法二：并查集 (Union-Find)

**费曼一句话：** 每个人头上顶个牌子写着「我的老大是谁」。两个人认识了，就让一个人的老大认另一个人的老大当老大。最后数有几个「终极老大」，就有几个圈子。

![并查集 parent 数组变化](/images/graph-theory/11_union_find_steps.png)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px  # 按秩合并
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True

# 使用
uf = UnionFind(8)
for u, v in [(0,1),(1,2),(0,2),(3,4),(5,6),(6,7)]:
    uf.union(u, v)
# 数连通分量 = 不同根的个数
print(len(set(uf.find(i) for i in range(8))))  # 输出: 3
```

**数据流追踪：**
```
初始: parent=[0,1,2,3,4,5,6,7] (每人是自己的老大)

union(0,1): parent=[0,0,2,3,4,5,6,7]  (1的老大→0)
union(1,2): find(1)=0, parent=[0,0,0,3,4,5,6,7]  (2的老大→0)
union(0,2): find(0)=0, find(2)=0, 同根跳过
union(3,4): parent=[0,0,0,3,3,5,6,7]
union(5,6): parent=[0,0,0,3,3,5,5,7]
union(6,7): parent=[0,0,0,3,3,5,5,5]

根集合: {0, 3, 5} → 3个连通分量
```

> 🎯 **DFS vs 并查集怎么选？** 静态图（边不会增加）→ DFS 更简单。动态图（边不断加入，需要实时查询连通性）→ 并查集，因为 union/find 接近 O(1)。

---

### 2.3 二分图判定（染色法）

**费曼一句话：** 把所有人分成两队，要求每条边连接的两个人必须在不同队。如果能分成功，就是二分图。方法很简单：BFS 一层层染色，红蓝交替，遇到矛盾就不是。

![二分图判定](/images/graph-theory/09_bipartite.png)

**关键定理：** 一个图是二分图 ⟺ 图中不存在奇数长度的环。

```python
from collections import deque

def is_bipartite(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    color = [-1] * n  # -1=未染色, 0=红, 1=蓝
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in graph[node]:
                if color[nb] == -1:
                    color[nb] = 1 - color[node]  # 染相反色
                    queue.append(nb)
                elif color[nb] == color[node]:
                    return False  # 同色相邻=矛盾!
    return True
```

**数据流追踪（二分图: 0-1, 0-3, 2-1, 2-3, 4-1, 4-3）：**
```
start=0: color[0]=0(红), queue=[0]
  pop 0: 邻居 1,3
    color[1]=-1 → 染1(蓝), queue=[1,3]
    color[3]=-1 → 染1(蓝)
  pop 1: 邻居 0,2,4
    color[0]=0 ≠ color[1]=1 → OK
    color[2]=-1 → 染0(红), queue=[3,2,4]
    color[4]=-1 → 染0(红)
  pop 3: 邻居 0,2,4
    color[0]=0 ≠ 1 → OK
    color[2]=0 ≠ 1 → OK
    color[4]=0 ≠ 1 → OK
  pop 2, pop 4: 邻居都已染色且无矛盾
→ 是二分图! 红={0,2,4}, 蓝={1,3}
```

**数据流追踪（非二分图: 0-1, 1-2, 2-0, 三角形）：**
```
start=0: color[0]=0(红)
  pop 0: 邻居 1,2
    color[1]=1(蓝), color[2]=1(蓝)
  pop 1: 邻居 0,2
    color[0]=0 ≠ 1 → OK
    color[2]=1 == color[1]=1 → 矛盾! return False
→ 不是二分图 (三角形=奇数环)
```

---

### 2.4 最短路径

#### 无权图：BFS 直接搞定

**费曼一句话：** 每条边长度都是 1，BFS 天然按层扩展，第一次到达就是最短路。

```python
from collections import deque

def shortest_path_bfs(graph, start, end):
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            return dist[end]
        for nb in graph[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return -1  # 不可达
```

#### 有权图：Dijkstra

**费曼一句话：** 贪心策略——每次从「已知最短距离的候选人」中挑最小的那个，确认它的最短路，然后用它去更新邻居。就像水从源头流出，总是先到最近的地方。

![Dijkstra 逐步过程](/images/graph-theory/10_dijkstra_steps.png)

```python
import heapq

def dijkstra(graph, start):
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue  # 过时的条目，跳过
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

**数据流追踪（图: 0→1:4, 0→2:1, 2→1:2, 1→3:1, 2→4:5, 3→4:3）：**
```
初始: dist={0:0}, heap=[(0,0)]

pop (0,0): 处理节点0
  → 邻居1: 0+4=4 < inf → dist={0:0,1:4}, push(4,1)
  → 邻居2: 0+1=1 < inf → dist={0:0,1:4,2:1}, push(1,2)
  heap=[(1,2),(4,1)]

pop (1,2): 处理节点2
  → 邻居1: 1+2=3 < 4 → dist[1]=3, push(3,1)  ← 松弛!
  → 邻居4: 1+5=6 < inf → dist[4]=6, push(6,4)
  heap=[(3,1),(4,1),(6,4)]

pop (3,1): 处理节点1
  → 邻居3: 3+1=4 < inf → dist[3]=4, push(4,3)
  heap=[(4,1),(4,3),(6,4)]

pop (4,1): d=4 > dist[1]=3 → 过时条目,跳过!

pop (4,3): 处理节点3
  → 邻居4: 4+3=7 > dist[4]=6 → 不更新
  heap=[(6,4)]

pop (6,4): 处理节点4, 无更新
  heap=[] → 结束

最终: dist = {0:0, 1:3, 2:1, 3:4, 4:6}
最短路径: 0→2→1→3, 0→2→4
```

> ⚠️ **Dijkstra 不能处理负权边！** 因为贪心假设「已确认的最短距离不会再变小」，负权边会打破这个假设。负权图用 Bellman-Ford。

> 🎯 **复杂度：** 用最小堆实现，时间 O((V+E)logV)，空间 O(V)。

---

## 📝 技巧篇小结

| 技巧 | 核心思想 | 数据结构 | 时间复杂度 |
|------|---------|---------|-----------|
| 拓扑排序(Kahn) | 不断删入度为0的节点 | 队列+入度数组 | O(V+E) |
| 拓扑排序(DFS) | 后序反转 | 递归栈+三色标记 | O(V+E) |
| 连通分量(DFS) | 从未访问节点出发DFS | visited数组 | O(V+E) |
| 连通分量(并查集) | union边, 数根 | parent数组 | O(Eα(V))≈O(E) |
| 二分图判定 | BFS染色, 检查矛盾 | color数组+队列 | O(V+E) |
| 最短路(无权) | BFS | 队列+dist | O(V+E) |
| 最短路(有权) | Dijkstra贪心 | 最小堆+dist | O((V+E)logV) |

---

## 第三章：实战篇 — LeetCode 真题拆解

---

### 3.1 LeetCode 200. 岛屿数量

**题意：** 给一个 `m×n` 的二维网格，`'1'` 是陆地，`'0'` 是水。计算岛屿数量（上下左右相连的陆地算一个岛）。

**费曼一句话：** 这就是求连通分量！每块相连的陆地是一个连通分量。从每个未访问的 `'1'` 出发做 DFS，把整块岛「淹掉」（标记为已访问），数你淹了几次。

**本质：** 网格就是图，每个格子是节点，上下左右是边。

```python
class Solution:
    def numIslands(self, grid):
        if not grid: return 0
        m, n = len(grid), len(grid[0])
        count = 0

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return
            grid[i][j] = '0'  # 淹掉，避免重复访问
            dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    dfs(i, j)
                    count += 1
        return count
```

**数据流追踪：**
```
输入:
  1 1 0 0 0
  1 1 0 0 0
  0 0 1 0 0
  0 0 0 1 1

i=0,j=0: grid[0][0]='1' → dfs淹掉整块
  dfs(0,0)→dfs(1,0)→dfs(1,1)→dfs(0,1)  全部变'0'
  count=1

i=2,j=2: grid[2][2]='1' → dfs淹掉
  count=2

i=3,j=3: grid[3][3]='1' → dfs(3,3)→dfs(3,4)
  count=3

结果: 3个岛屿
```

> 🎯 **核心技巧：** 直接修改原数组当 visited，省空间。如果不能改原数组，用 visited 集合。

---

### 3.2 LeetCode 207/210. 课程表（拓扑排序）

**题意：** 207 — 给 n 门课和先修关系，判断能否修完所有课。210 — 返回一个合法的修课顺序。

**费曼一句话：** 先修关系就是有向边，能修完 = 没有环 = 能拓扑排序。

```python
# 207. 能否修完
class Solution:
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for a, b in prerequisites:  # b→a (先修b才能上a)
            graph[b].append(a)
            indegree[a] += 1
        queue = deque(i for i in range(numCourses) if indegree[i] == 0)
        count = 0
        while queue:
            node = queue.popleft()
            count += 1
            for nb in graph[node]:
                indegree[nb] -= 1
                if indegree[nb] == 0:
                    queue.append(nb)
        return count == numCourses
```

```python
# 210. 返回修课顺序 (只需把count换成result列表)
class Solution:
    def findOrder(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for a, b in prerequisites:
            graph[b].append(a)
            indegree[a] += 1
        queue = deque(i for i in range(numCourses) if indegree[i] == 0)
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for nb in graph[node]:
                indegree[nb] -= 1
                if indegree[nb] == 0:
                    queue.append(nb)
        return result if len(result) == numCourses else []
```

**数据流追踪（n=4, prereqs=[[1,0],[2,0],[3,1],[3,2]]）：**
```
图: 0→1, 0→2, 1→3, 2→3
indegree: [0,1,1,2]
queue=[0], result=[]

pop 0 → result=[0], indegree[1]=0→入队, indegree[2]=0→入队
pop 1 → result=[0,1], indegree[3]=1
pop 2 → result=[0,1,2], indegree[3]=0→入队
pop 3 → result=[0,1,2,3]

len=4=n → 合法! 顺序: [0,1,2,3]
```

---

### 3.3 LeetCode 133. 克隆图

**题意：** 给一个无向连通图的某个节点引用，返回该图的深拷贝。

**费曼一句话：** 遍历原图，每遇到一个节点就「克隆」一份。用哈希表记录「原节点→克隆节点」的映射，避免重复克隆（也就是 visited 的作用）。

```python
class Solution:
    def cloneGraph(self, node):
        if not node: return None
        cloned = {node: Node(node.val)}
        queue = deque([node])
        while queue:
            curr = queue.popleft()
            for nb in curr.neighbors:
                if nb not in cloned:
                    cloned[nb] = Node(nb.val)
                    queue.append(nb)
                cloned[curr].neighbors.append(cloned[nb])
        return cloned[node]
```

**数据流追踪（图: 1-2, 1-4, 2-3, 3-4）：**
```
cloned={1:Node(1)}, queue=[1]

pop 1: 邻居 [2,4]
  2 不在cloned → cloned[2]=Node(2), 入队
  4 不在cloned → cloned[4]=Node(4), 入队
  clone[1].neighbors = [clone[2], clone[4]]

pop 2: 邻居 [1,3]
  1 已在cloned → 跳过创建
  3 不在cloned → cloned[3]=Node(3), 入队
  clone[2].neighbors = [clone[1], clone[3]]

pop 4: 邻居 [1,3]
  clone[4].neighbors = [clone[1], clone[3]]

pop 3: 邻居 [2,4]
  clone[3].neighbors = [clone[2], clone[4]]

返回 cloned[1] → 完整的深拷贝图
```

> 🎯 **核心技巧：** `cloned` 字典同时充当 visited 和映射表，一石二鸟。

---

### 3.4 LeetCode 785. 判断二分图

**题意：** 给一个邻接表表示的无向图，判断是否是二分图。

**费曼一句话：** 直接套染色法模板。BFS 红蓝交替染，遇到矛盾就不是。

```python
class Solution:
    def isBipartite(self, graph):
        n = len(graph)
        color = [-1] * n
        for i in range(n):
            if color[i] != -1: continue
            color[i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in graph[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False
        return True
```

> 🎯 **注意：** 图可能不连通，所以外层要遍历所有节点作为起点。

---

### 3.5 LeetCode 743. 网络延迟时间（Dijkstra）

**题意：** n 个节点的有向加权图，从节点 k 发信号，返回所有节点收到信号的最短时间。如果有节点收不到，返回 -1。

**费曼一句话：** 从 k 出发跑 Dijkstra，求到所有节点的最短距离，取最大值就是答案。

```python
class Solution:
    def networkDelayTime(self, times, n, k):
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))
        dist = {}
        heap = [(0, k)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in dist: continue
            dist[u] = d
            for v, w in graph[u]:
                if v not in dist:
                    heapq.heappush(heap, (d + w, v))
        return max(dist.values()) if len(dist) == n else -1
```

**数据流追踪（times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2）：**
```
graph: {2:[(1,1),(3,1)], 3:[(4,1)]}
heap=[(0,2)]

pop (0,2): dist={2:0}, push (1,1),(1,3)
pop (1,1): dist={2:0,1:1}
pop (1,3): dist={2:0,1:1,3:1}, push (2,4)
pop (2,4): dist={2:0,1:1,3:1,4:2}

len(dist)=4=n → max(0,1,1,2) = 2
```

---

### 3.6 LeetCode 994. 腐烂的橘子（多源 BFS）

**题意：** 网格中 0=空，1=新鲜橘子，2=腐烂橘子。每分钟腐烂橘子会感染上下左右的新鲜橘子。返回所有橘子腐烂的最短时间，不可能则返回 -1。

**费曼一句话：** 多个腐烂橘子同时开始扩散，就像同时往池塘里扔好几块石头。把所有腐烂橘子一起放进队列作为起点，然后标准 BFS 层序扩展，层数就是时间。

```python
class Solution:
    def orangesRotting(self, grid):
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1
        if fresh == 0: return 0
        minutes = 0
        while queue:
            minutes += 1
            for _ in range(len(queue)):
                x, y = queue.popleft()
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx_, ny_ = x+dx, y+dy
                    if 0<=nx_<m and 0<=ny_<n and grid[nx_][ny_]==1:
                        grid[nx_][ny_] = 2
                        fresh -= 1
                        queue.append((nx_, ny_))
            if fresh == 0: return minutes
        return -1
```

**数据流追踪：**
```
输入:
  2 1 1
  1 1 0
  0 1 1

初始: queue=[(0,0)], fresh=7

Minute 1: 扩散(0,0)→感染(0,1),(1,0)
  grid:  2 2 1    fresh=5
         2 1 0
         0 1 1

Minute 2: 扩散(0,1),(1,0)→感染(0,2),(1,1)
  grid:  2 2 2    fresh=3
         2 2 0
         0 1 1

Minute 3: 扩散(0,2),(1,1)→感染(2,1)
  grid:  2 2 2    fresh=2
         2 2 0
         0 2 1

Minute 4: 扩散(2,1)→感染(2,2)
  grid:  2 2 2    fresh=0 → return 4
         2 2 0
         0 2 2
```

> 🎯 **多源 BFS 的关键：** 把所有源点一起入队，而不是对每个源点分别 BFS。这样时间复杂度还是 O(m×n)。

---

## 第四章：Anki 闪记卡片

以下 15 张卡片可直接导入 Anki（Q/A 格式）。

---

**Q1:** 图的邻接表和邻接矩阵，刷题默认用哪个？为什么？
**A1:** 邻接表。空间 O(V+E) 更省，遍历邻居 O(degree) 更快。90% 的图题都用它。

---

**Q2:** DFS 用什么数据结构？一句话描述它的策略。
**A2:** 栈（递归调用栈或显式栈）。策略：一条路走到黑，撞墙再回头。

---

**Q3:** BFS 用什么数据结构？为什么它能求无权图最短路？
**A3:** 队列（FIFO）。因为 BFS 按层扩展，第一次到达某节点时走的边数一定最少。

---

**Q4:** BFS 中 visited 应该在入队时标记还是出队时？为什么？
**A4:** 入队时！出队时标记会导致同一节点被多次入队，浪费时间空间。

---

**Q5:** 拓扑排序的前提条件是什么？
**A5:** 图必须是有向无环图（DAG）。有环则无法拓扑排序。

---

**Q6:** Kahn's BFS 拓扑排序的核心操作是什么？
**A6:** 不断找入度为 0 的节点，弹出并删除它的出边，使邻居入度减 1。重复直到队列空。

---

**Q7:** 如何用 DFS 做拓扑排序？
**A7:** DFS 后序记录完成顺序，最后反转。用三色标记（白/灰/黑）检测环：遇到灰色节点说明有环。

---

**Q8:** 什么是连通分量？怎么数？
**A8:** 图中互相可达的最大节点集合。方法：遍历所有节点，每次从未访问节点出发 DFS/BFS，计数 +1。

---

**Q9:** 并查集的两个核心优化是什么？
**A9:** 路径压缩（find 时直接指向根）+ 按秩合并（矮树挂到高树下）。使 union/find 接近 O(1)。

---

**Q10:** 如何判断一个图是否是二分图？
**A10:** BFS 染色法：红蓝交替染色，如果相邻节点同色则不是二分图。等价条件：不存在奇数环。

---

**Q11:** Dijkstra 算法的核心贪心策略是什么？
**A11:** 每次从未确认的节点中选距离最小的，确认其最短路，然后用它松弛邻居。用最小堆实现。

---

**Q12:** Dijkstra 为什么不能处理负权边？
**A12:** 贪心假设「已确认的最短距离不会再变小」，负权边会打破这个假设。负权图用 Bellman-Ford。

---

**Q13:** LeetCode 200 岛屿数量的本质是什么？
**A13:** 求网格图的连通分量数。每次从未访问的 '1' 出发 DFS 淹掉整块岛，计数 +1。

---

**Q14:** 多源 BFS 和普通 BFS 的区别？
**A14:** 多源 BFS 把所有源点同时入队作为第 0 层，然后正常层序扩展。典型题：994 腐烂的橘子。

---

**Q15:** DFS 和 BFS 怎么选？
**A15:** 要最短路 → BFS。要遍历所有路径/连通性/拓扑排序 → DFS。不确定 → 都行，选顺手的。

---

## 第五章：LeetCode 刷题路线图

### 阶段一：入门（掌握 DFS/BFS 模板）

| 题号 | 题目 | 核心技巧 | 难度 |
|------|------|---------|------|
| 200 | 岛屿数量 | DFS/BFS 连通分量 | Medium |
| 733 | 图像渲染 | Flood Fill (DFS) | Easy |
| 695 | 岛屿的最大面积 | DFS + 计数 | Medium |
| 994 | 腐烂的橘子 | 多源 BFS | Medium |
| 542 | 01 矩阵 | 多源 BFS 求最短距离 | Medium |
| 133 | 克隆图 | BFS + 哈希映射 | Medium |

### 阶段二：进阶（拓扑排序 + 二分图 + 并查集）

| 题号 | 题目 | 核心技巧 | 难度 |
|------|------|---------|------|
| 207 | 课程表 | Kahn's 拓扑排序 | Medium |
| 210 | 课程表 II | 拓扑排序输出序列 | Medium |
| 785 | 判断二分图 | BFS 染色 | Medium |
| 886 | 可能的二分法 | 二分图变体 | Medium |
| 547 | 省份数量 | 并查集/DFS 连通分量 | Medium |
| 684 | 冗余连接 | 并查集检测环 | Medium |

### 阶段三：硬核（最短路 + 综合应用）

| 题号 | 题目 | 核心技巧 | 难度 |
|------|------|---------|------|
| 743 | 网络延迟时间 | Dijkstra | Medium |
| 787 | K 站中转最便宜航班 | Bellman-Ford/BFS | Medium |
| 1091 | 二进制矩阵最短路径 | BFS 8方向 | Medium |
| 127 | 单词接龙 | BFS + 状态图 | Hard |
| 329 | 矩阵中的最长递增路径 | DFS + 记忆化 | Hard |
| 1192 | 查找集群内的关键连接 | Tarjan 求桥 | Hard |

---

## 第六章：算法选择决策树

拿到一道图题，脑子里应该跑这个决策流程：

```
题目给了图 →
│
├─ 求最短路？
│   ├─ 无权图 → BFS
│   ├─ 有权图(非负权) → Dijkstra
│   └─ 有负权 → Bellman-Ford
│
├─ 判断能否完成/有无环？
│   └─ 有向图 → 拓扑排序 (Kahn's BFS)
│
├─ 求连通分量/岛屿数量？
│   ├─ 静态图 → DFS
│   └─ 动态加边 → 并查集
│
├─ 判断二分图？
│   └─ BFS 染色法
│
├─ 求所有路径/排列组合？
│   └─ DFS + 回溯
│
└─ 不确定？
    └─ 先试 BFS（如果需要最短/最少），否则 DFS
```

**复杂度速查表：**

| 算法 | 时间 | 空间 | 适用条件 |
|------|------|------|---------|
| BFS | O(V+E) | O(V) | 无权最短路、层序遍历 |
| DFS | O(V+E) | O(V) | 连通性、回溯、拓扑排序 |
| Dijkstra(堆) | O((V+E)logV) | O(V) | 非负权最短路 |
| Bellman-Ford | O(VE) | O(V) | 有负权、检测负环 |
| Kahn's拓扑 | O(V+E) | O(V) | DAG排序、环检测 |
| 并查集 | O(Eα(V)) | O(V) | 动态连通性 |
| Floyd-Warshall | O(V³) | O(V²) | 全源最短路(小图) |

**面试/竞赛中的选择优先级：**

1. 看到「最短」「最少步数」「最近」→ 先想 BFS
2. 看到「所有路径」「排列」「组合」→ 先想 DFS + 回溯
3. 看到「先修课」「依赖关系」「顺序」→ 先想拓扑排序
4. 看到「分组」「两队」「对立」→ 先想二分图染色
5. 看到「连通」「岛屿」「省份」→ 先想 DFS 或并查集
6. 看到「带权最短路」→ Dijkstra（确认无负权）

---

## 总结：图论知识地图

```
                    图论与搜索
                       │
          ┌────────────┼────────────┐
          │            │            │
       表示法        遍历算法      经典应用
       │            │            │
  ┌────┼────┐   ┌───┼───┐   ┌───┼────┬────┬────┐
  │    │    │   │       │   │   │    │    │    │
邻接  邻接  边  DFS    BFS  拓扑 连通  二分  最短
矩阵  表   列表 (栈)  (队列) 排序 分量  图   路径
                              │    │    │    │
                          Kahn's DFS  染色 Dijkstra
                          DFS后序 并查集    BFS(无权)
```

**一句话总结全文：** 图论的核心就两件事——怎么存图（邻接表），怎么遍历图（DFS/BFS）。所有高级应用都是在遍历的基础上加点料。

---

> 本文约 9300 字，包含 11 张可视化图解、15 张 Anki 闪记卡片、18 道 LeetCode 刷题路线。
> 如有错误或建议，欢迎指正。