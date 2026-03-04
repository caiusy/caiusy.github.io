---
title: 链表与指针完全图解：从内存本质到 LeetCode 实战
date: 2026-02-22 10:54:38
tags:
  - 数据结构
  - 链表
  - 指针
  - LeetCode
  - 算法
categories:
  - 算法与数据结构
mathjax: true
---

> **费曼说**：如果你不能用简单的语言解释一个东西，说明你还没真正理解它。
> 本文用「内存视角」带你从零建立链表直觉，配合 10+ 张原创图解和 LeetCode 经典题，让你彻底搞懂指针到底在干什么。

<!-- more -->

---

## 一、指针的本质：一个存地址的变量

### 1.1 先忘掉"指针"这个词

很多人一听"指针"就怕。其实指针就是一个**存地址的变量**，仅此而已。

想象你住在一栋公寓楼里：
- 每个房间有一个**门牌号**（内存地址）
- 房间里放着**东西**（数据）
- 你手里有一张**纸条**，上面写着某个门牌号 —— 这张纸条就是指针

```
纸条上写着: 0x1010
你去 0x1010 房间一看，里面放着数字 7
```

就这么简单。指针不神秘，它就是一个"写着地址的纸条"。

### 1.2 内存模型图解

![内存模型：指针本质](/images/linkedlist/01_memory_pointer.png)

看这张图：
- `int a = 42` 住在地址 `0x1000`，房间里放着 `42`
- `int b = 7` 住在地址 `0x1010`，房间里放着 `7`
- `int *p = &b` 住在地址 `0x1008`，房间里放着 `0x1010` —— 也就是 b 的地址

当你写 `*p` 的时候，计算机做了两步：
1. 先看 p 里存的地址：`0x1010`
2. 去 `0x1010` 取值：`7`

这就是**解引用（dereference）**，没有任何魔法。

### 1.3 用 C/Python 感受指针

**C 语言版（显式指针）：**

```c
int b = 7;
int *p = &b;    // p 存的是 b 的地址
printf("%d", *p); // 输出 7 —— 通过地址找到值
*p = 100;        // 通过地址修改值
printf("%d", b);  // 输出 100 —— b 被改了！
```

**Python 版（隐式引用）：**

```python
# Python 里一切都是引用（指针的高级包装）
a = [1, 2, 3]
b = a          # b 和 a 指向同一个列表对象
b.append(4)
print(a)       # [1, 2, 3, 4] —— a 也变了！因为 b 和 a 是同一个地址
```

> **费曼笔记**：指针 = 地址。`*p` = 去那个地址看看。`&x` = 告诉我 x 的地址。三句话说完了。

---

## 二、链表节点：val + next 的组合拳

### 2.1 为什么需要链表？

数组的问题：
- 插入/删除要**搬移大量元素**，O(n)
- 大小固定（静态数组）或需要重新分配（动态数组）

链表的优势：
- 插入/删除只需要**改几个指针**，O(1)
- 大小完全动态，用多少分配多少

代价：
- 不能随机访问（没有 `arr[i]`），只能从头遍历
- 每个节点多存一个指针，空间开销更大

### 2.2 节点结构图解

![链表节点结构](/images/linkedlist/02_node_structure.png)

每个节点就两样东西：
- **val**：存数据（蓝色框）
- **next**：存下一个节点的地址（橙色框）

最后一个节点的 next 指向 **NULL**（空），表示链表结束。

### 2.3 代码定义

**Python 版：**

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 创建链表: 1 -> 3 -> 5 -> 7
node4 = ListNode(7)
node3 = ListNode(5, node4)
node2 = ListNode(3, node3)
head = ListNode(1, node2)
```

**C++ 版：**

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};
```

### 2.4 遍历：链表的基本功

```python
def traverse(head):
    curr = head
    while curr:
        print(curr.val, end=" -> ")
        curr = curr.next
    print("NULL")

# 输出: 1 -> 3 -> 5 -> 7 -> NULL
```

**数据流追踪：**

```
Step 1: curr = head       → curr.val = 1, curr.next = node2
Step 2: curr = curr.next  → curr.val = 3, curr.next = node3
Step 3: curr = curr.next  → curr.val = 5, curr.next = node4
Step 4: curr = curr.next  → curr.val = 7, curr.next = None
Step 5: curr = None       → 循环结束
```

> **费曼笔记**：遍历链表就像寻宝游戏 —— 每个宝箱里有一个宝物（val）和一张纸条告诉你下一个宝箱在哪（next）。纸条写着"没了"（NULL）就结束。


---

## 三、链表的核心操作：插入与删除

### 3.1 插入节点：先接后断

插入的核心原则：**先让新节点接上后面，再让前面接上新节点**。顺序反了就断链了。

![插入节点图解](/images/linkedlist/03_insert_node.png)

**在 node(3) 后面插入 node(9)：**

```python
def insert_after(prev_node, new_val):
    new_node = ListNode(new_val)
    new_node.next = prev_node.next   # Step 1: 新节点先接上后面
    prev_node.next = new_node        # Step 2: 前面再接上新节点
```

**数据流追踪（插入 9 到 3 后面）：**

```
初始状态: 1 -> 3 -> 5 -> 7 -> NULL
                ↑ prev_node

Step 1: new_node(9).next = prev_node.next
        → new_node(9).next = node(5)
        此时: 1 -> 3 -> 5 -> 7 -> NULL
                        ↑
              new(9) ---┘

Step 2: prev_node.next = new_node
        → node(3).next = node(9)
        此时: 1 -> 3 -> 9 -> 5 -> 7 -> NULL  ✓
```

> ⚠️ **经典错误**：如果先执行 Step 2，`prev_node.next` 就被覆盖了，你再也找不到原来的 node(5)，链表断裂！

### 3.2 删除节点：跳过它

删除不需要真的"销毁"节点，只需要让前一个节点**跳过**它。

![删除节点图解](/images/linkedlist/04_delete_node.png)

```python
def delete_next(prev_node):
    if prev_node.next:
        prev_node.next = prev_node.next.next  # 直接跳过下一个节点
```

**数据流追踪（删除 3）：**

```
初始: 1 -> 3 -> 5 -> 7 -> NULL
      ↑prev

prev.next = prev.next.next
→ node(1).next = node(5)

结果: 1 -> 5 -> 7 -> NULL
      node(3) 被跳过，无人引用，等待垃圾回收
```

### 3.3 时间复杂度对比

| 操作 | 数组 | 链表 |
|------|------|------|
| 按索引访问 | O(1) | O(n) |
| 头部插入 | O(n) | O(1) |
| 中间插入（已知位置） | O(n) | O(1) |
| 尾部插入 | O(1)* | O(n)** |
| 删除（已知位置） | O(n) | O(1) |

> *摊还复杂度 **需要遍历到尾部

> **费曼笔记**：插入 = 先接后断（顺序不能反）。删除 = 让前一个直接指向后一个（跳过中间人）。所有链表操作的本质都是在**改 next 指针**。

---

## 四、虚拟头节点（Dummy Head）：消灭特殊情况

### 4.1 痛点：头节点的特殊处理

删除链表中值为 x 的节点时，如果要删的恰好是头节点怎么办？

```python
# 没有 dummy head 的写法 —— 丑陋且易错
def remove_val_ugly(head, val):
    # 特殊处理：头节点就是目标
    while head and head.val == val:
        head = head.next
    
    # 常规处理：中间节点
    curr = head
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head
```

两段逻辑，两种处理方式，容易出 bug。

### 4.2 解法：加一个假的头

![虚拟头节点](/images/linkedlist/08_dummy_head.png)

在真正的头节点前面加一个 **dummy 节点**，这样所有节点（包括原来的头）都有"前一个节点"，逻辑统一。

```python
# 有 dummy head 的写法 —— 干净统一
def remove_val_clean(head, val):
    dummy = ListNode(0)
    dummy.next = head
    
    curr = dummy
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next  # 统一的删除逻辑
        else:
            curr = curr.next
    
    return dummy.next  # 返回真正的头
```

> **费曼笔记**：Dummy head 就像在队伍最前面放一个"占位的人"。有了它，不管删谁，逻辑都一样：`prev.next = prev.next.next`。这是链表题的**第一个必备技巧**。


---

## 五、双指针技巧：链表的瑞士军刀

双指针是链表题的核心武器，分三种用法：

### 5.1 快慢指针：找中点

**问题**：不知道链表长度，如何一次遍历找到中间节点？

**直觉**：两个人同时从起点出发，一个走一步，一个走两步。快的到终点时，慢的刚好在中间。

```python
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next        # 慢指针走 1 步
        fast = fast.next.next   # 快指针走 2 步
    return slow  # slow 就是中点

# 数据流追踪: 1 -> 2 -> 3 -> 4 -> 5
# Step 0: slow=1, fast=1
# Step 1: slow=2, fast=3
# Step 2: slow=3, fast=5
# fast.next=None, 停！slow=3 就是中点 ✓
```

**应用场景**：
- LeetCode 876：链表的中间节点
- 归并排序链表的分割步骤
- 判断回文链表的前半段

### 5.2 间距指针：删除倒数第 N 个节点（LeetCode 19）

**问题**：一次遍历删除倒数第 N 个节点。

**直觉**：让 fast 先走 N 步，然后 slow 和 fast 一起走。fast 到终点时，slow 刚好在倒数第 N+1 个位置。

![删除倒数第N个节点](/images/linkedlist/09_remove_nth.png)

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    slow = fast = dummy
    
    # fast 先走 n+1 步
    for _ in range(n + 1):
        fast = fast.next
    
    # 一起走到底
    while fast:
        slow = slow.next
        fast = fast.next
    
    # slow.next 就是要删的节点
    slow.next = slow.next.next
    return dummy.next
```

**数据流追踪（删除倒数第 2 个，即 node(4)）：**

```
链表: D -> 1 -> 2 -> 3 -> 4 -> 5 -> NULL
n=2

Phase 1: fast 先走 3 步 (n+1)
  fast = D -> 1 -> 2 -> 3
  slow = D

Phase 2: 一起走
  Step 1: slow=1, fast=4
  Step 2: slow=2, fast=5
  Step 3: slow=3, fast=NULL → 停！

Delete: slow.next = slow.next.next
  → node(3).next = node(5)
  
结果: 1 -> 2 -> 3 -> 5 -> NULL ✓
```

> **为什么走 n+1 步而不是 n 步？** 因为我们需要停在目标节点的**前一个**，才能执行 `prev.next = prev.next.next`。


---

## 六、反转链表：链表题的灵魂（LeetCode 206）

反转链表是链表题的**绝对核心**，至少有 10 道 LeetCode 题是它的变体。不掌握这个，后面的题全卡住。

### 6.1 直觉：逐个掰方向

想象一列火车车厢，每节车厢的挂钩都朝右。反转就是把每个挂钩**掰向左边**。

![反转链表图解](/images/linkedlist/05_reverse_list.png)

### 6.2 迭代法（推荐掌握）

核心思路：用三个指针 `prev`、`curr`、`next_temp`，逐个翻转。

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next   # 1. 先存下一个（不然断了找不到）
        curr.next = prev        # 2. 掰方向：当前节点指向前面
        prev = curr             # 3. prev 前进
        curr = next_temp        # 4. curr 前进
    return prev  # prev 就是新的头
```

**逐步数据流追踪：**

```
初始: prev=NULL, curr=1->2->3->NULL

=== 第 1 轮 ===
next_temp = 2 (保存)
curr(1).next = NULL (prev)     → 1->NULL
prev = 1
curr = 2
状态: NULL<-1  2->3->NULL
      prev↑   curr↑

=== 第 2 轮 ===
next_temp = 3
curr(2).next = 1 (prev)        → 2->1->NULL
prev = 2
curr = 3
状态: NULL<-1<-2  3->NULL
            prev↑ curr↑

=== 第 3 轮 ===
next_temp = NULL
curr(3).next = 2 (prev)        → 3->2->1->NULL
prev = 3
curr = NULL
状态: NULL<-1<-2<-3
                  prev↑  curr=NULL → 结束

返回 prev = node(3)，即新头
结果: 3 -> 2 -> 1 -> NULL ✓
```

### 6.3 递归法（理解即可）

```python
def reverseList_recursive(head):
    # base case: 空链表或只有一个节点
    if not head or not head.next:
        return head
    
    # 递归反转后面的部分
    new_head = reverseList_recursive(head.next)
    
    # 把下一个节点的 next 指回自己
    head.next.next = head
    head.next = None
    
    return new_head
```

**递归展开追踪（1->2->3->NULL）：**

```
调用栈展开:
  reverse(1) → reverse(2) → reverse(3) → return 3 (base case)

回溯:
  reverse(2): head=2, new_head=3
    head.next.next = head → node(3).next = node(2)  → 3->2
    head.next = None      → node(2).next = None     → 3->2->NULL
    return 3

  reverse(1): head=1, new_head=3
    head.next.next = head → node(2).next = node(1)  → 3->2->1
    head.next = None      → node(1).next = None     → 3->2->1->NULL
    return 3

最终: 3->2->1->NULL ✓
```

> **费曼笔记**：反转链表的本质就是三步舞 —— 存、掰、走。`next_temp` 存后路，`curr.next = prev` 掰方向，然后两个指针一起往前走。记住这个节奏，所有反转变体都是它的延伸。


---

## 七、环形链表：快慢指针的经典战场（LeetCode 141 / 142）

### 7.1 问题：链表有没有环？

如果链表的某个节点的 next 指向了前面的某个节点，就形成了**环**。遍历永远不会到 NULL，程序会死循环。

![环检测图解](/images/linkedlist/06_cycle_detection.png)

### 7.2 Floyd 判圈算法（龟兔赛跑）

**直觉**：操场跑步，快的人（fast）和慢的人（slow）同时出发。如果操场是环形的，快的人一定会**从后面追上**慢的人。如果操场是直线的，快的人直接跑到终点，永远不会相遇。

```python
# LeetCode 141: 判断是否有环
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True   # 相遇了 → 有环
    return False          # fast 到终点了 → 无环
```

**数据流追踪（有环：1->2->3->4->5->3）：**

```
Step 0: slow=1, fast=1
Step 1: slow=2, fast=3
Step 2: slow=3, fast=5
Step 3: slow=4, fast=4  ← 相遇！有环 ✓
```

### 7.3 进阶：找环的入口（LeetCode 142）

不仅要判断有没有环，还要找到**环从哪里开始**。

**数学推导（简化版）：**

设：
- 起点到环入口距离 = `a`
- 环入口到相遇点距离 = `b`
- 环的长度 = `c`

相遇时：
- slow 走了 `a + b` 步
- fast 走了 `a + b + n*c` 步（多绕了 n 圈）
- fast 速度是 slow 的 2 倍：`2(a+b) = a + b + n*c`
- 化简：`a = n*c - b = (n-1)*c + (c-b)`

**结论**：从**起点**和**相遇点**同时出发，各走一步，它们会在**环入口**相遇！

```python
# LeetCode 142: 找环的入口
def detectCycle(head):
    slow = fast = head
    
    # Phase 1: 找相遇点
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # 无环
    
    # Phase 2: 找入口
    slow = head  # slow 回到起点
    while slow != fast:
        slow = slow.next
        fast = fast.next  # 注意：这里 fast 也只走一步！
    
    return slow  # 相遇点就是环入口
```

**数据流追踪（1->2->3->4->5->3，环入口=3）：**

```
Phase 1 (找相遇点):
  slow: 1→2→3→4
  fast: 1→3→5→4
  相遇于 node(4)

Phase 2 (找入口):
  slow 从 head(1) 出发, fast 从 node(4) 出发
  Step 1: slow=2, fast=5
  Step 2: slow=3, fast=3  ← 相遇！环入口 = node(3) ✓
```

> **费曼笔记**：判环 = 龟兔赛跑，追上了就有环。找入口 = 数学魔术，从起点和相遇点同速出发，再次相遇就是入口。记住这两个 phase 就够了。


---

## 八、合并两个有序链表（LeetCode 21）

### 8.1 问题

给你两个升序链表，合并成一个升序链表。

![合并有序链表](/images/linkedlist/07_merge_sorted.png)

### 8.2 思路：拉拉链

想象你手里有两副扑克牌，都已经从小到大排好了。你要把它们合成一副：

1. 比较两副牌顶部的牌
2. 小的那张抽出来放到新牌堆
3. 重复，直到某一副用完
4. 把剩下的那副直接接上

```python
def mergeTwoLists(l1, l2):
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
    
    curr.next = l1 or l2  # 接上剩余部分
    return dummy.next
```

**数据流追踪（L1: 1->3->5, L2: 2->4->6）：**

```
初始: dummy -> ?    l1=1, l2=2

Step 1: 1 < 2 → pick 1    dummy->1    l1=3, l2=2
Step 2: 3 > 2 → pick 2    dummy->1->2    l1=3, l2=4
Step 3: 3 < 4 → pick 3    dummy->1->2->3    l1=5, l2=4
Step 4: 5 > 4 → pick 4    dummy->1->2->3->4    l1=5, l2=6
Step 5: 5 < 6 → pick 5    dummy->1->2->3->4->5    l1=NULL, l2=6
Step 6: l1=NULL → 接上 l2  dummy->1->2->3->4->5->6

结果: 1->2->3->4->5->6 ✓
```

### 8.3 复杂度

- 时间：O(m+n)，每个节点恰好被访问一次
- 空间：O(1)，只用了几个指针（不算 dummy）

> **费曼笔记**：合并有序链表 = 拉拉链。两边比大小，小的先接上。用完一边，另一边直接粘上去。dummy head 让你不用操心"第一个节点接到哪"。


---

## 九、LeetCode 实战进阶

### 9.1 回文链表（LeetCode 234）

**问题**：判断链表是否是回文（正读反读一样）。

**思路**：三步走 —— 找中点 → 反转后半段 → 逐一比较。

```python
def isPalindrome(head):
    # Step 1: 快慢指针找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: 反转后半段
    prev = None
    curr = slow
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    # Step 3: 比较前半段和反转后的后半段
    left, right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True
```

**数据流追踪（1->2->3->2->1）：**

```
Step 1 找中点:
  slow: 1→2→3  fast: 1→3→NULL
  中点 = 3

Step 2 反转后半段 (3->2->1):
  反转后: 1->2->3

Step 3 比较:
  left=1, right=1 → 相等 ✓
  left=2, right=2 → 相等 ✓
  left=3, right=3 → 相等 ✓
  right=NULL → 结束，是回文 ✓
```

> 这道题把前面学的**快慢指针**和**反转链表**组合起来了。链表题就是这样，基础技巧的排列组合。

### 9.2 相交链表（LeetCode 160）

**问题**：两个链表可能在某个节点汇合，找到那个交点。

**直觉**：两个人走不同的路，路的终点相同。如果 A 走完自己的路再走 B 的路，B 走完自己的路再走 A 的路，它们走的总距离相同，一定会在交点相遇。

```python
def getIntersectionNode(headA, headB):
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB  # a 走完就去 B 的起点
        b = b.next if b else headA  # b 走完就去 A 的起点
    return a  # 相遇点就是交点（或者都是 None = 无交点）
```

**为什么有效？**

```
A 的路: a1 → a2 → c1 → c2 → c3     (独有2 + 公共3 = 5)
B 的路: b1 → b2 → b3 → c1 → c2 → c3 (独有3 + 公共3 = 6)

A 走完走 B: a1→a2→c1→c2→c3→b1→b2→b3→c1  (5+3=8步到c1)
B 走完走 A: b1→b2→b3→c1→c2→c3→a1→a2→c1  (6+2=8步到c1)

两人都在第 8 步到达 c1 → 相遇！
```

> **费曼笔记**：浪漫算法 —— 你走过我来时的路，我走过你来时的路，我们终将在交汇处相遇。数学上就是 `a + c + b = b + c + a`。

### 9.3 两数相加（LeetCode 2）

**问题**：两个链表表示两个数（逆序存储），求它们的和。

```
输入: (2->4->3) + (5->6->4)
代表: 342 + 465 = 807
输出: 7->0->8
```

```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    carry = 0
    
    while l1 or l2 or carry:
        val = carry
        if l1:
            val += l1.val
            l1 = l1.next
        if l2:
            val += l2.val
            l2 = l2.next
        
        carry = val // 10
        curr.next = ListNode(val % 10)
        curr = curr.next
    
    return dummy.next
```

**数据流追踪：**

```
Step 1: 2+5+0=7, carry=0 → 创建 node(7)
Step 2: 4+6+0=10, carry=1 → 创建 node(0)
Step 3: 3+4+1=8, carry=0 → 创建 node(8)
Step 4: 全部为空且 carry=0 → 结束

结果: 7->0->8 ✓ (即 807)
```


---

## 十、进阶：排序链表（LeetCode 148）

### 10.1 问题

对链表进行排序，要求 O(n log n) 时间复杂度。

**思路**：归并排序天然适合链表 —— 找中点分割 + 递归排序 + 合并有序链表。前面学的技巧全用上了。

```python
def sortList(head):
    # base case
    if not head or not head.next:
        return head
    
    # Step 1: 快慢指针找中点，断开
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None  # 断开！
    
    # Step 2: 递归排序两半
    left = sortList(head)
    right = sortList(mid)
    
    # Step 3: 合并（复用 mergeTwoLists）
    dummy = ListNode(0)
    curr = dummy
    while left and right:
        if left.val <= right.val:
            curr.next = left
            left = left.next
        else:
            curr.next = right
            right = right.next
        curr = curr.next
    curr.next = left or right
    return dummy.next
```

**递归分解追踪（4->2->1->3）：**

```
sortList(4->2->1->3)
├── 分割: [4->2] 和 [1->3]
│   ├── sortList(4->2)
│   │   ├── 分割: [4] 和 [2]
│   │   └── 合并: 2->4
│   └── sortList(1->3)
│       ├── 分割: [1] 和 [3]
│       └── 合并: 1->3
└── 合并 [2->4] 和 [1->3]: 1->2->3->4 ✓
```

> **费曼笔记**：链表排序 = 找中点（快慢指针）+ 分两半（断开）+ 各自排序（递归）+ 合并（拉拉链）。四个基础操作的组合拳，没有新东西。

---

## 十一、K 个一组翻转链表（LeetCode 25）—— 链表终极 Boss

### 11.1 问题

每 k 个节点一组进行翻转。如果最后不足 k 个，保持原样。

```
输入: 1->2->3->4->5, k=3
输出: 3->2->1->4->5
```

### 11.2 思路拆解

1. 先数 k 个节点，不够就不翻
2. 翻转这 k 个节点（就是 LC 206 的子问题）
3. 递归处理剩余部分
4. 把翻转后的尾巴接上递归结果

```python
def reverseKGroup(head, k):
    # Step 1: 检查是否有 k 个节点
    curr = head
    count = 0
    while curr and count < k:
        curr = curr.next
        count += 1
    if count < k:
        return head  # 不足 k 个，不翻转
    
    # Step 2: 翻转前 k 个
    prev = None
    curr = head
    for _ in range(k):
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    
    # Step 3: head 现在是翻转后的尾巴，接上递归结果
    head.next = reverseKGroup(curr, k)
    
    return prev  # prev 是翻转后的新头
```

**数据流追踪（1->2->3->4->5, k=3）：**

```
第一次调用: head=1, 数到 k=3 ✓
  翻转 [1,2,3]: prev=3->2->1
  head(1).next = reverseKGroup(4->5, 3)
    第二次调用: head=4, 数到 2 < 3
    → 不翻转，返回 4->5
  head(1).next = 4->5
  
最终: 3->2->1->4->5 ✓
```

> **费曼笔记**：K 组翻转 = 数够 k 个就翻，翻完递归处理剩下的，不够就不动。核心还是反转链表，只是加了"分段"和"递归"。


---

## 十二、链表题型全景图：模式识别

做了这么多题，我们来提炼**模式**。链表题万变不离其宗，核心就这几个套路：

### 12.1 五大核心技巧

| 技巧 | 适用场景 | 代表题目 |
|------|----------|----------|
| **虚拟头节点** | 需要删除/修改头节点的场景 | LC 203, 21, 19, 82 |
| **快慢指针** | 找中点、判环、找环入口 | LC 876, 141, 142, 234 |
| **间距指针** | 倒数第 N 个、窗口滑动 | LC 19, 61 |
| **反转链表** | 翻转全部/部分/K组 | LC 206, 92, 25, 234 |
| **合并链表** | 有序合并、分治 | LC 21, 23, 148 |

### 12.2 解题决策树

```
拿到链表题 →
├── 需要删除节点？ → 用 dummy head
├── 需要找中点/判环？ → 用快慢指针
├── 需要倒数第 N 个？ → 用间距指针
├── 需要翻转？ → 用 prev/curr/next 三指针
├── 需要合并？ → 用 dummy + 比较
└── 需要排序？ → 归并排序（找中点+分割+合并）
```

### 12.3 常见陷阱清单

**1. 空指针访问**
```python
# 错误：没检查 curr 是否为 None
while curr.next:  # 如果 curr 是 None，直接崩溃

# 正确：
while curr and curr.next:
```

**2. 断链丢失**
```python
# 错误：先断后接，丢失后续节点
prev.next = new_node      # 原来的 prev.next 丢了！
new_node.next = ???        # 找不到了

# 正确：先接后断
new_node.next = prev.next  # 先保存
prev.next = new_node       # 再修改
```

**3. 忘记更新指针**
```python
# 错误：删除后忘记移动
while curr.next:
    if curr.next.val == target:
        curr.next = curr.next.next
        # 忘了：这里不应该 curr = curr.next！
        # 因为新的 curr.next 可能还是 target
    else:
        curr = curr.next
```

**4. 返回值错误**
```python
# 错误：返回 head（可能已经被删了）
# 正确：用 dummy head，返回 dummy.next
```


---

## 十三、费曼终极总结：用一句话说清每个概念

> 费曼检验：如果你能用一句话向一个 10 岁小孩解释清楚，你才真正理解了。

| 概念 | 一句话解释 |
|------|-----------|
| **指针** | 一张写着门牌号的纸条，拿着它就能找到对应的房间 |
| **链表节点** | 一个盒子，里面装着一个东西和一张纸条（指向下一个盒子） |
| **遍历** | 寻宝游戏：打开盒子，看纸条，去下一个盒子，直到纸条写着"没了" |
| **插入** | 新人插队：先让新人拉住后面的人，再让前面的人拉住新人 |
| **删除** | 踢出队伍：让前面的人直接拉住后面的人，中间那个自然就出局了 |
| **虚拟头** | 在队伍最前面放一个假人，这样不管踢谁，操作都一样 |
| **快慢指针** | 两人赛跑，快的走两步慢的走一步，快的到终点时慢的在中间 |
| **判环** | 操场跑步，快的追上慢的说明是环形跑道 |
| **反转** | 把每节火车车厢的挂钩方向掰过来 |
| **合并** | 两副排好的扑克牌，每次比顶上的，小的先出 |

### 链表的本质

**链表的所有操作，归根结底就是在做一件事：改 next 指针。**

- 插入 = 改两个 next
- 删除 = 改一个 next
- 反转 = 改所有 next 的方向
- 合并 = 按条件决定 next 接谁

理解了这一点，你就理解了链表的全部。

### 从数组到链表的思维转换

```
数组思维：我要访问第 i 个元素 → arr[i] → O(1)
链表思维：我要从头走到第 i 个 → 走 i 步 → O(n)

数组思维：插入一个元素 → 后面全部搬家 → O(n)
链表思维：改两个指针 → O(1)

数组思维：数据在内存中连续排列
链表思维：数据散落在内存各处，用指针串起来
```

> **终极费曼笔记**：数组是一排紧挨着的储物柜（连续内存），你知道编号就能直接打开。链表是散落各处的宝箱（离散内存），每个宝箱里有一张纸条告诉你下一个在哪。各有优劣，选择取决于你更需要"快速找到第 N 个"还是"快速插入删除"。


---

## 十四、闪记卡片（Flashcards）

> 以下卡片适合用 Anki 导入，采用 Q/A 格式。每张卡片对应一个核心知识点。

---

### 卡片 1：指针是什么？

**Q**: 用一句话解释指针。

**A**: 指针是一个变量，它存储的不是数据本身，而是数据所在的**内存地址**。就像一张写着门牌号的纸条。`*p` = 去那个地址看看，`&x` = 告诉我 x 的地址。

---

### 卡片 2：链表节点的结构

**Q**: 链表节点包含哪两个部分？

**A**: `val`（存数据）和 `next`（存下一个节点的地址）。最后一个节点的 next 指向 NULL。

---

### 卡片 3：链表插入的正确顺序

**Q**: 在 prev 后面插入 new_node，两步操作的顺序是什么？为什么？

**A**: 
1. `new_node.next = prev.next`（先接后面）
2. `prev.next = new_node`（再断前面）

顺序不能反！如果先执行 Step 2，`prev.next` 被覆盖，原来的后续节点就丢了。

---

### 卡片 4：虚拟头节点的作用

**Q**: 什么时候需要 dummy head？它解决什么问题？

**A**: 当操作可能涉及**头节点的删除或修改**时使用。它在真正的 head 前面加一个假节点，使所有节点（包括 head）都有前驱节点，逻辑统一为 `prev.next = prev.next.next`。最后返回 `dummy.next`。

---

### 卡片 5：快慢指针找中点

**Q**: 如何一次遍历找到链表中点？

**A**: slow 走 1 步，fast 走 2 步。当 `fast` 或 `fast.next` 为 NULL 时，`slow` 就在中点。原理：fast 速度是 slow 的 2 倍，fast 到终点时 slow 走了一半。

---

### 卡片 6：Floyd 判圈算法

**Q**: 如何判断链表是否有环？如何找到环的入口？

**A**: 
- **判环**：快慢指针，相遇则有环。
- **找入口**：相遇后，让 slow 回到 head，两个指针都走 1 步，再次相遇点就是入口。数学原理：`a = (n-1)c + (c-b)`。

---

### 卡片 7：反转链表的三步舞

**Q**: 迭代反转链表的核心操作是什么？

**A**: 三个指针 `prev, curr, next_temp`，循环执行：
1. `next_temp = curr.next`（存后路）
2. `curr.next = prev`（掰方向）
3. `prev = curr; curr = next_temp`（前进）

循环结束后 `prev` 是新头。

---

### 卡片 8：合并两个有序链表

**Q**: 合并两个有序链表的核心思路？

**A**: 用 dummy head，每次比较两个链表头部，小的接上去，对应指针前进。某一方用完后，直接把另一方剩余部分接上。时间 O(m+n)，空间 O(1)。

---

### 卡片 9：间距指针删除倒数第 N 个

**Q**: 如何一次遍历删除倒数第 N 个节点？

**A**: fast 先走 N+1 步，然后 slow 和 fast 一起走。fast 到 NULL 时，`slow.next` 就是要删的节点。用 dummy head 处理删头节点的边界情况。

---

### 卡片 10：链表 vs 数组

**Q**: 链表和数组的核心区别？各自的优势场景？

**A**: 
| | 数组 | 链表 |
|---|---|---|
| 内存 | 连续 | 离散 |
| 随机访问 | O(1) | O(n) |
| 插入删除 | O(n) | O(1) |
| 适用场景 | 频繁查询 | 频繁增删 |

---

### 卡片 11：链表排序用什么算法？

**Q**: 对链表排序，最优方案是什么？

**A**: **归并排序**。步骤：快慢指针找中点 → 断开 → 递归排序两半 → 合并有序链表。时间 O(n log n)，空间 O(log n)（递归栈）。不用快排，因为链表不支持随机访问。

---

### 卡片 12：K 个一组翻转的思路

**Q**: LeetCode 25 "K 个一组翻转链表"的核心思路？

**A**: 
1. 数 k 个节点，不够就不翻
2. 翻转这 k 个（LC 206 子问题）
3. 原来的 head 变成尾巴，接上递归处理剩余部分的结果
4. 返回翻转后的新头 prev


---

## 十五、LeetCode 刷题路线图

按难度递进，建议按顺序刷：

### 第一阶段：基础操作（Easy）

| # | 题目 | 核心技巧 | 难度 |
|---|------|----------|------|
| 206 | 反转链表 | prev/curr/next 三指针 | ⭐ |
| 21 | 合并两个有序链表 | dummy head + 比较 | ⭐ |
| 141 | 环形链表 | 快慢指针 | ⭐ |
| 203 | 移除链表元素 | dummy head | ⭐ |
| 876 | 链表的中间节点 | 快慢指针 | ⭐ |
| 83 | 删除排序链表中的重复元素 | 遍历 + 跳过 | ⭐ |
| 160 | 相交链表 | 双指针走对方的路 | ⭐ |

### 第二阶段：技巧组合（Medium）

| # | 题目 | 核心技巧 | 难度 |
|---|------|----------|------|
| 2 | 两数相加 | 遍历 + 进位 | ⭐⭐ |
| 19 | 删除链表的倒数第 N 个节点 | 间距指针 + dummy | ⭐⭐ |
| 24 | 两两交换链表中的节点 | 反转变体 | ⭐⭐ |
| 142 | 环形链表 II | Floyd 找入口 | ⭐⭐ |
| 234 | 回文链表 | 快慢 + 反转 + 比较 | ⭐⭐ |
| 148 | 排序链表 | 归并排序 | ⭐⭐ |
| 82 | 删除排序链表中的重复元素 II | dummy + 跳过所有重复 | ⭐⭐ |
| 92 | 反转链表 II | 区间反转 | ⭐⭐ |
| 61 | 旋转链表 | 成环再断开 | ⭐⭐ |
| 138 | 复制带随机指针的链表 | 哈希表 / 交织复制 | ⭐⭐ |

### 第三阶段：终极挑战（Hard）

| # | 题目 | 核心技巧 | 难度 |
|---|------|----------|------|
| 25 | K 个一组翻转链表 | 分段反转 + 递归 | ⭐⭐⭐ |
| 23 | 合并 K 个升序链表 | 分治 / 最小堆 | ⭐⭐⭐ |


---

## 十六、附录：完整链表工具类

以下是一个可以直接用于 LeetCode 本地调试的工具类：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def build_list(arr):
    """从数组创建链表: [1,2,3] -> 1->2->3->NULL"""
    dummy = ListNode(0)
    curr = dummy
    for v in arr:
        curr.next = ListNode(v)
        curr = curr.next
    return dummy.next

def to_array(head):
    """链表转数组: 1->2->3->NULL -> [1,2,3]"""
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res

def print_list(head):
    """打印链表: 1 -> 2 -> 3 -> NULL"""
    parts = []
    while head:
        parts.append(str(head.val))
        head = head.next
    print(" -> ".join(parts) + " -> NULL")

# 使用示例
head = build_list([1, 2, 3, 4, 5])
print_list(head)  # 1 -> 2 -> 3 -> 4 -> 5 -> NULL
```

---

## 写在最后

链表看似简单，实则是**指针操作的训练场**。掌握了链表，你就掌握了：

- **指针思维**：理解引用、地址、间接访问
- **边界处理**：空指针、头尾节点、单元素
- **画图能力**：复杂指针操作必须画图，脑子里转不过来
- **递归直觉**：链表天然适合递归思考

**最重要的一条建议**：做链表题时，一定要**画图**。把每一步的指针变化画出来，比在脑子里想快 10 倍。

> "What I cannot create, I do not understand." — Richard Feynman

---

*本文包含 10+ 张原创图解、12 张闪记卡片、20+ 道 LeetCode 题目解析。如果对你有帮助，欢迎收藏和分享。*
