---
title: 🪟 LeetCode 150 - 滑动窗口专题
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
description: LeetCode 面试 150 题之滑动窗口专题，含图解、代码模板、记忆口诀
type: note
note_type: algorithm
difficulty: intermediate
review_status: reviewing
---
# 🪟 滑动窗口专题 (4题)

> 🎯 **核心技巧**：右扩左缩、哈希计数、条件收缩

---

## 🗺️ 滑动窗口核心思想

```
┌─────────────────────────────────────────────────────────────┐
│                    滑动窗口工作原理                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Step 1: 右边界扩张                                         │
│   ┌───┬───┬───┬───┬───┬───┬───┐                            │
│   │ a │ b │ c │ d │ e │ f │ g │                            │
│   └───┴───┴───┴───┴───┴───┴───┘                            │
│     L           R ──────▶                                   │
│     └─────┬─────┘                                           │
│         window                                              │
│                                                             │
│   Step 2: 满足条件时左边界收缩                                │
│   ┌───┬───┬───┬───┬───┬───┬───┐                            │
│   │ a │ b │ c │ d │ e │ f │ g │                            │
│   └───┴───┴───┴───┴───┴───┴───┘                            │
│     L ──▶       R                                           │
│         └───┬───┘                                           │
│           window (收缩后)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 滑动窗口万能模板

```python
def sliding_window(s):
    from collections import defaultdict
    
    window = defaultdict(int)  # 窗口内容计数
    left = 0
    result = 0  # 或者其他结果变量
    
    for right in range(len(s)):
        # ① 右边界扩张：将 s[right] 加入窗口
        c = s[right]
        window[c] += 1
        
        # ② 判断是否需要收缩（根据题目条件）
        while 需要收缩的条件:
            # 将 s[left] 移出窗口
            d = s[left]
            window[d] -= 1
            left += 1
        
        # ③ 更新结果（根据题目要求）
        result = max(result, right - left + 1)
    
    return result
```

---

## 1️⃣ LC 209. 长度最小的子数组 🟡

### 题目描述
找出和 ≥ target 的最短连续子数组长度。

### 🎨 图解思路

```
nums = [2, 3, 1, 2, 4, 3], target = 7

窗口滑动过程：
┌─────────────────────────────────────────┐
│ [2]                  sum=2  < 7  扩张   │
│ [2,3]                sum=5  < 7  扩张   │
│ [2,3,1]              sum=6  < 7  扩张   │
│ [2,3,1,2]            sum=8  ≥ 7  记录4  │
│   [3,1,2]            sum=6  < 7  扩张   │
│   [3,1,2,4]          sum=10 ≥ 7  记录4  │
│     [1,2,4]          sum=7  ≥ 7  记录3  │
│       [2,4]          sum=6  < 7  扩张   │
│       [2,4,3]        sum=9  ≥ 7  记录3  │
│         [4,3]        sum=7  ≥ 7  记录2 ✓│
└─────────────────────────────────────────┘

最小长度 = 2
```

### 💻 代码实现

```python
def minSubArrayLen(target: int, nums: list) -> int:
    left = 0
    window_sum = 0
    min_len = float('inf')
    
    for right in range(len(nums)):
        # 扩张：加入右边元素
        window_sum += nums[right]
        
        # 收缩：满足条件时尝试缩小窗口
        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1
    
    return min_len if min_len != float('inf') else 0
```

### 🧠 记忆口诀
> **"够了就缩，不够就扩"**

---

## 2️⃣ LC 3. 无重复字符的最长子串 🟡

### 题目描述
找出不含重复字符的最长子串长度。

### 🎨 图解思路

```
s = "abcabcbb"

窗口滑动过程：
┌─────────────────────────────────────────┐
│ [a]         无重复  len=1               │
│ [a,b]       无重复  len=2               │
│ [a,b,c]     无重复  len=3 ✓             │
│ [a,b,c,a]   有重复! 收缩                │
│   [b,c,a]   无重复  len=3               │
│   [b,c,a,b] 有重复! 收缩                │
│     [c,a,b] 无重复  len=3               │
│     ...                                 │
└─────────────────────────────────────────┘

最长无重复子串长度 = 3
```

### 💻 代码实现

```python
def lengthOfLongestSubstring(s: str) -> int:
    window = {}  # 记录字符出现次数
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        c = s[right]
        window[c] = window.get(c, 0) + 1
        
        # 有重复字符时收缩
        while window[c] > 1:
            d = s[left]
            window[d] -= 1
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### 🔥 优化版本（记录位置）

```python
def lengthOfLongestSubstring(s: str) -> int:
    char_index = {}  # 记录字符最后出现的位置
    left = 0
    max_len = 0
    
    for right, c in enumerate(s):
        # 如果字符在窗口内出现过，直接跳到重复位置之后
        if c in char_index and char_index[c] >= left:
            left = char_index[c] + 1
        
        char_index[c] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

### 🧠 记忆口诀
> **"重复就跳过，记录最长度"**

---

## 3️⃣ LC 30. 串联所有单词的子串 🔴

### 题目描述
找出字符串中所有是 `words` 中所有单词串联结果的起始索引。

### 🎨 图解思路

```
s = "barfoothefoobarman"
words = ["foo", "bar"]  (每个长度为3)

串联结果可能是 "foobar" 或 "barfoo"（长度6）

检查每个可能的起始位置：
位置0: "barfoo" ✓ (bar + foo)
位置3: "foothe" ✗
位置6: "thefoo" ✗
位置9: "foobar" ✓ (foo + bar)
...

结果: [0, 9]
```

### 💻 代码实现

```python
def findSubstring(s: str, words: list) -> list:
    if not s or not words:
        return []
    
    from collections import Counter
    
    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    word_freq = Counter(words)
    result = []
    
    # 只需要检查 word_len 种起始偏移
    for offset in range(word_len):
        left = offset
        window = Counter()
        count = 0  # 窗口内有效单词数
        
        for right in range(offset, len(s) - word_len + 1, word_len):
            word = s[right:right + word_len]
            
            if word in word_freq:
                window[word] += 1
                count += 1
                
                # 如果某单词超出需要的数量，收缩左边界
                while window[word] > word_freq[word]:
                    left_word = s[left:left + word_len]
                    window[left_word] -= 1
                    count -= 1
                    left += word_len
                
                # 找到一个有效的串联
                if count == word_count:
                    result.append(left)
            else:
                # 遇到不在 words 中的单词，重置窗口
                window.clear()
                count = 0
                left = right + word_len
    
    return result
```

### 🧠 记忆口诀
> **"固定单词长，滑动找串联"**

---

## 4️⃣ LC 76. 最小覆盖子串 🔴

### 题目描述
找出 s 中包含 t 所有字符的最小子串。

### 🎨 图解思路

```
s = "ADOBECODEBANC", t = "ABC"

需要: A:1, B:1, C:1

┌─────────────────────────────────────────────┐
│ 窗口: [A]DOBECODEBANC     缺BC   扩张        │
│ 窗口: [ADOBEC]ODEBANC     满足!  记录6 收缩  │
│ 窗口: [DOBEC]ODEBANC      缺A    扩张        │
│ 窗口: [DOBECODEBA]NC      满足!  记录10 收缩 │
│ 窗口: [CODEBA]NC          满足!  记录6 收缩  │
│ 窗口: [ODEBA]NC           缺C    扩张        │
│ 窗口: [ODEBANC]           满足!  记录7 收缩  │
│ 窗口: [BANC]              满足!  记录4 ✓     │
└─────────────────────────────────────────────┘

最小覆盖子串 = "BANC"
```

### 💻 代码实现

```python
def minWindow(s: str, t: str) -> str:
    from collections import Counter
    
    need = Counter(t)       # 需要的字符及数量
    window = Counter()      # 窗口内的字符
    
    have = 0                # 已满足的字符种类数
    need_count = len(need)  # 需要满足的字符种类数
    
    result = ""
    min_len = float('inf')
    left = 0
    
    for right in range(len(s)):
        # 扩张：加入右边字符
        c = s[right]
        window[c] += 1
        
        # 如果该字符数量正好满足需求
        if c in need and window[c] == need[c]:
            have += 1
        
        # 收缩：当所有字符都满足时
        while have == need_count:
            # 更新结果
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]
            
            # 移出左边字符
            d = s[left]
            if d in need and window[d] == need[d]:
                have -= 1
            window[d] -= 1
            left += 1
    
    return result
```

### 🧠 记忆口诀
> **"扩到满足，缩到不够"**

---

## 📊 本章总结

### 滑动窗口分类

| 类型 | 特点 | 典型题目 |
|------|------|----------|
| **可变窗口求最小** | 满足条件就收缩 | 209, 76 |
| **可变窗口求最大** | 不满足条件才收缩 | 3 |
| **固定窗口** | 窗口大小固定 | 30 |

### 滑动窗口思维导图

```
        满足条件？
           │
     ┌─────┴─────┐
     ▼           ▼
    是           否
     │           │
     ▼           ▼
  求最小？     求最大？
     │           │
     ▼           ▼
   收缩        扩张
  更新答案    继续扩张
```

### 🧠 全章记忆口诀

```
最无串覆四道题
滑动窗口巧解析
右扩左缩是关键
满足条件再收缩

最 - 长度最小的子数组 (209)
无 - 无重复字符的最长子串 (3)
串 - 串联所有单词的子串 (30)
覆 - 最小覆盖子串 (76)
```

---

> 📖 **下一篇**：[链表专题](/2026/01/18/leetcode-150-linked-list/)

