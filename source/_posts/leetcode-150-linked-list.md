---
title: 🔗 LeetCode 150 - 链表专题
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
  - 链表
description: LeetCode 面试 150 题之链表专题，含图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 🔗 链表专题 (11题)

> 🎯 **核心技巧**：虚拟头节点、快慢指针、链表反转、合并链表

---

## 🗺️ 链表核心操作图解

```
┌─────────────────────────────────────────────────────────────┐
│                     链表基本操作                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  遍历:  [1] → [2] → [3] → [4] → None                       │
│          ↑                                                  │
│         cur (cur = cur.next)                               │
│                                                             │
│  插入:  [1] → [X] → [2]   (先接后断)                        │
│              ↗   ↘                                         │
│                                                             │
│  删除:  [1] ──────→ [3]   (跨过中间节点)                    │
│              ╳[2]                                          │
│                                                             │
│  反转:  [1] ← [2] ← [3] ← [4]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 链表万能技巧

### 技巧1: 虚拟头节点 (Dummy Head)

```python
# 避免处理头节点的特殊情况
dummy = ListNode(0)
dummy.next = head
# ... 操作链表
return dummy.next
```

### 技巧2: 快慢指针

```python
# 找中点、判断环、找环入口
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow 就是中点
```

### 技巧3: 链表反转

```python
def reverse(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
```

---

## 1️⃣ LC 141. 环形链表 🟢

### 题目描述
判断链表中是否有环。

### 🎨 图解思路

```
快慢指针：如果有环，快指针一定会追上慢指针

    ┌───────────────┐
    ▼               │
[1] → [2] → [3] → [4]
        ↑     ↑
       slow  fast

快指针每次走2步，慢指针每次走1步
相对速度为1，一定会在环内相遇
```

### 💻 代码实现

```python
def hasCycle(head: ListNode) -> bool:
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            return True
    
    return False
```

### 🧠 记忆口诀
> **"快慢追逐，相遇有环"**

---

## 2️⃣ LC 2. 两数相加 🟡

### 题目描述
两个逆序存储的链表相加，返回结果链表。

### 🎨 图解思路

```
   2 → 4 → 3  (表示 342)
 + 5 → 6 → 4  (表示 465)
 ─────────────
   7 → 0 → 8  (表示 807)

从头到尾逐位相加，注意进位！

Step 1: 2 + 5 = 7, 进位0
Step 2: 4 + 6 = 10, 写0进1  
Step 3: 3 + 4 + 1 = 8, 进位0
```

### 💻 代码实现

```python
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    carry = 0
    
    while l1 or l2 or carry:
        # 取值（链表可能长度不同）
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        
        # 计算和与进位
        total = val1 + val2 + carry
        carry = total // 10
        
        # 创建新节点
        curr.next = ListNode(total % 10)
        curr = curr.next
        
        # 移动指针
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    
    return dummy.next
```

### 🧠 记忆口诀
> **"逐位相加，别忘进位"**

---

## 3️⃣ LC 21. 合并两个有序链表 🟢

### 题目描述
将两个升序链表合并为一个新的升序链表。

### 🎨 图解思路

```
l1: 1 → 2 → 4
l2: 1 → 3 → 4

比较头节点，小的接上去：

dummy → 1 → 1 → 2 → 3 → 4 → 4
        ↑   ↑   ↑   ↑   ↑   ↑
       l1  l2  l1  l2  l1  l2
```

### 💻 代码实现

```python
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    # 接上剩余部分
    curr.next = l1 if l1 else l2
    
    return dummy.next
```

### 🧠 记忆口诀
> **"比较取小，剩余直接接"**

---

## 4️⃣ LC 138. 随机链表的复制 🟡

### 题目描述
深拷贝带有随机指针的链表。

### 🎨 图解思路

```
方法：哈希表存储映射关系

原链表:  [1] → [2] → [3]
          ↓     ↓     ↓ (random)
         [3]   [1]   None

Step 1: 创建所有新节点，建立 old→new 映射
Step 2: 连接 next 和 random 指针
```

### 💻 代码实现

```python
def copyRandomList(head: 'Node') -> 'Node':
    if not head:
        return None
    
    # 建立映射: 旧节点 → 新节点
    old_to_new = {}
    
    # 第一遍：创建所有新节点
    curr = head
    while curr:
        old_to_new[curr] = Node(curr.val)
        curr = curr.next
    
    # 第二遍：连接 next 和 random
    curr = head
    while curr:
        new_node = old_to_new[curr]
        new_node.next = old_to_new.get(curr.next)
        new_node.random = old_to_new.get(curr.random)
        curr = curr.next
    
    return old_to_new[head]
```

### 🔥 O(1) 空间解法

```python
def copyRandomList(head: 'Node') -> 'Node':
    if not head:
        return None
    
    # Step 1: 在每个节点后插入复制节点
    # A → A' → B → B' → C → C'
    curr = head
    while curr:
        new_node = Node(curr.val, curr.next)
        curr.next = new_node
        curr = new_node.next
    
    # Step 2: 设置 random 指针
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    
    # Step 3: 拆分链表
    dummy = Node(0)
    new_curr = dummy
    curr = head
    while curr:
        new_curr.next = curr.next
        new_curr = new_curr.next
        curr.next = curr.next.next
        curr = curr.next
    
    return dummy.next
```

### 🧠 记忆口诀
> **"哈希映射，两遍搞定"**

---

## 5️⃣ LC 92. 反转链表 II 🟡

### 题目描述
反转链表的第 left 到第 right 个节点。

### 🎨 图解思路

```
原链表: 1 → 2 → 3 → 4 → 5, left=2, right=4

Step 1: 找到 left 前一个节点
        1 → [2 → 3 → 4] → 5
        ↑    └─反转区间─┘
       prev

Step 2: 反转中间部分
        1    2 ← 3 ← 4    5
        ↑    ↓         ↑
       prev  └─────────┘

Step 3: 重新连接
        1 → 4 → 3 → 2 → 5
```

### 💻 代码实现

```python
def reverseBetween(head: ListNode, left: int, right: int) -> ListNode:
    dummy = ListNode(0, head)
    prev = dummy
    
    # 移动到 left 前一个位置
    for _ in range(left - 1):
        prev = prev.next
    
    # 反转 [left, right] 区间
    curr = prev.next
    for _ in range(right - left):
        # 把 curr.next 插到 prev 后面
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node
    
    return dummy.next
```

### 🧠 记忆口诀
> **"头插法反转，一次遍历"**

---

## 6️⃣ LC 25. K 个一组翻转链表 🔴

### 题目描述
每 k 个节点一组进行翻转。

### 🎨 图解思路

```
链表: 1 → 2 → 3 → 4 → 5, k = 2

第1组: [1,2] → 反转 → [2,1]
第2组: [3,4] → 反转 → [4,3]  
第3组: [5] → 不足k个，保持原样

结果: 2 → 1 → 4 → 3 → 5
```

### 💻 代码实现

```python
def reverseKGroup(head: ListNode, k: int) -> ListNode:
    # 检查是否有 k 个节点
    def get_kth(node, k):
        while node and k > 0:
            node = node.next
            k -= 1
        return node
    
    # 反转链表
    def reverse(head, tail):
        prev = tail.next
        curr = head
        while prev != tail:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp
        return tail, head  # 新的头和尾
    
    dummy = ListNode(0, head)
    prev_group = dummy
    
    while True:
        # 找到这一组的尾节点
        kth = get_kth(prev_group, k)
        if not kth:
            break
        
        next_group = kth.next
        
        # 反转这一组
        head, tail = prev_group.next, kth
        new_head, new_tail = reverse(head, tail)
        
        # 连接
        prev_group.next = new_head
        new_tail.next = next_group
        
        prev_group = new_tail
    
    return dummy.next
```

### 🧠 记忆口诀
> **"够k就翻，不够就留"**

---

## 7️⃣ LC 19. 删除链表的倒数第 N 个节点 🟡

### 题目描述
删除链表的倒数第 n 个节点。

### 🎨 图解思路

```
快慢指针，快指针先走 n 步

链表: 1 → 2 → 3 → 4 → 5, n = 2

Step 1: fast 先走 2 步
        1 → 2 → 3 → 4 → 5
        ↑       ↑
       slow    fast

Step 2: 同时移动直到 fast 到末尾
        1 → 2 → 3 → 4 → 5 → None
                ↑       ↑
               slow    fast

Step 3: 删除 slow.next
        1 → 2 → 3 ────→ 5
```

### 💻 代码实现

```python
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    slow = fast = dummy
    
    # fast 先走 n+1 步
    for _ in range(n + 1):
        fast = fast.next
    
    # 同时移动
    while fast:
        slow = slow.next
        fast = fast.next
    
    # 删除 slow.next
    slow.next = slow.next.next
    
    return dummy.next
```

### 🧠 记忆口诀
> **"快先走n步，同行到末尾"**

---

## 8️⃣ LC 82. 删除排序链表中的重复元素 II 🟡

### 题目描述
删除所有重复的节点，只保留原始链表中没有重复出现的数字。

### 🎨 图解思路

```
1 → 2 → 3 → 3 → 4 → 4 → 5

检测重复并删除整组：
1 → 2 → [3 → 3] → [4 → 4] → 5
        删除      删除

结果: 1 → 2 → 5
```

### 💻 代码实现

```python
def deleteDuplicates(head: ListNode) -> ListNode:
    dummy = ListNode(0, head)
    prev = dummy
    
    while prev.next:
        curr = prev.next
        
        # 检测是否有重复
        if curr.next and curr.val == curr.next.val:
            # 跳过所有重复节点
            while curr.next and curr.val == curr.next.val:
                curr = curr.next
            prev.next = curr.next  # 删除整组
        else:
            prev = prev.next
    
    return dummy.next
```

### 🧠 记忆口诀
> **"见重复全删，不重复才留"**

---

## 9️⃣ LC 61. 旋转链表 🟡

### 题目描述
将链表每个节点向右移动 k 个位置。

### 🎨 图解思路

```
1 → 2 → 3 → 4 → 5, k = 2

Step 1: 连成环
        1 → 2 → 3 → 4 → 5
        ↑               │
        └───────────────┘

Step 2: 找到新的断点 (n - k % n)
        新头是第 5-2=3 个节点之后
        
Step 3: 在正确位置断开
        4 → 5 → 1 → 2 → 3
```

### 💻 代码实现

```python
def rotateRight(head: ListNode, k: int) -> ListNode:
    if not head or not head.next or k == 0:
        return head
    
    # 计算长度并找到尾节点
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    
    # 实际需要移动的步数
    k = k % length
    if k == 0:
        return head
    
    # 找到新的尾节点（第 length - k 个）
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    
    # 重新连接
    new_head = new_tail.next
    new_tail.next = None
    tail.next = head
    
    return new_head
```

### 🧠 记忆口诀
> **"先成环，再断开"**

---

## 🔟 LC 86. 分隔链表 🟡

### 题目描述
将链表按值 x 分成两部分：小于 x 的在前，大于等于 x 的在后。

### 🎨 图解思路

```
1 → 4 → 3 → 2 → 5 → 2, x = 3

分成两个链表：
小于3: 1 → 2 → 2
≥3:    4 → 3 → 5

合并: 1 → 2 → 2 → 4 → 3 → 5
```

### 💻 代码实现

```python
def partition(head: ListNode, x: int) -> ListNode:
    # 两个虚拟头节点
    small_dummy = ListNode(0)
    large_dummy = ListNode(0)
    small = small_dummy
    large = large_dummy
    
    while head:
        if head.val < x:
            small.next = head
            small = small.next
        else:
            large.next = head
            large = large.next
        head = head.next
    
    # 连接两个链表
    large.next = None  # 防止成环
    small.next = large_dummy.next
    
    return small_dummy.next
```

### 🧠 记忆口诀
> **"分两队，再合并"**

---

## 1️⃣1️⃣ LC 146. LRU 缓存 🟡

### 题目描述
实现 LRU (最近最少使用) 缓存机制。

### 🎨 图解思路

```
使用双向链表 + 哈希表

双向链表（按使用时间排序）:
head ⇄ [最近用] ⇄ [次近用] ⇄ ... ⇄ [最久未用] ⇄ tail

哈希表: key → 链表节点

get/put 操作:
1. 通过哈希表 O(1) 找到节点
2. 移动到链表头部（最近使用）
3. 超容量时删除链表尾部
```

### 💻 代码实现

```python
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}  # key → node
        self.capacity = capacity
        
        # 虚拟头尾节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _remove(self, node):
        """从链表中删除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _add_to_head(self, node):
        """添加到链表头部"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _move_to_head(self, node):
        """移动到头部"""
        self._remove(node)
        self._add_to_head(node)
    
    def _remove_tail(self):
        """删除尾部节点"""
        node = self.tail.prev
        self._remove(node)
        return node
    
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)
        return node.value
    
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            
            if len(self.cache) > self.capacity:
                tail = self._remove_tail()
                del self.cache[tail.key]
```

### 🧠 记忆口诀
> **"双链表记顺序，哈希表快查找"**

---

## 📊 本章总结

### 链表技巧速查表

| 技巧 | 使用场景 | 典型题目 |
|------|----------|----------|
| 虚拟头节点 | 可能修改头节点 | 21, 82, 86 |
| 快慢指针 | 找中点/判环 | 141, 19 |
| 反转链表 | 翻转操作 | 92, 25 |
| 哈希表辅助 | 复杂指针关系 | 138, 146 |
| 双指针 | 合并/分隔 | 21, 86 |

### 🧠 全章记忆口诀

```
环加合复反转组
删倒删重旋分缓

环 - 环形链表 (141)
加 - 两数相加 (2)
合 - 合并有序链表 (21)
复 - 复制随机链表 (138)
反 - 反转链表 II (92)
组 - K个一组翻转 (25)
删倒 - 删除倒数第N个 (19)
删重 - 删除重复元素 II (82)
旋 - 旋转链表 (61)
分 - 分隔链表 (86)
缓 - LRU缓存 (146)
```

---

> 📖 **下一篇**：[二叉树专题](/2026/01/18/leetcode-150-binary-tree/)

