---
title: "从一个 Token 理解 DeepSeek DSA：Lightning Indexer、Top-K 与 MLA 压缩缓存"
date: 2026-10-23 09:00:00
categories:
  - 大模型
  - 模型架构
tags:
  - DeepSeek
  - DSA
  - Lightning Indexer
  - MLA
  - Sparse Attention
type: article
difficulty: advanced
mathjax: true
description: "用一个 token 的完整推理链路讲清 Lightning Indexer、动态 Gate、Top-K 与 MLA 两套缓存，附交互演示和成本估算。"
review_status: published
---

DSA 的关键链路只有两步：**Lightning Indexer 用小 Key 扫描全部历史并选出 Top-K，Sparse MLA 再从压缩缓存中读取这些位置**。这篇文章从 token 9 的一次查询出发，逐个讲清 Query、共享 Key、动态 Gate、两套 Cache 与成本边界。

👉 [打开交互演示：自己调整 Gate、Top-K 与上下文长度](/dsa-lab/)

<!-- more -->



> 这篇文章不是从论文公式开始，而是从我真正学习 DSA 时卡住的几个问题开始：
>
> - `W_K` 是共享的，那 `k_t` 为什么又说不共享？
> - Head weights 到底是共享参数，还是每个 token 自己的值？
> - 为什么 Lightning Indexer 每一步必须用**当前 token 的 Gate**？
> - 当前 token 的 `k_t` 当下好像没用，为什么还要计算？
> - 历史 token 是不是要缓存 `h`，然后每次重新算 Key？
> - Top-K 选完以后，是不是还要访问 MLA 的压缩缓存？
> - Lightning Indexer 自己扫完整历史，那它真的省计算吗？
>
> 把这些问题理清之后，DSA 的推理链路其实非常干净。

> **阅读范围**：DeepSeek-V3.2 技术报告与 V3.2-Exp 官方 reference 实现，聚焦 Lightning Indexer 的推理链路与训练对齐，不展开 MoE 与后训练。所有配图为机制示意，按报告/官方代码重绘，收益注明比较条件；教学例子的数值人为设定，不代表实测日志。
>
> **公式分级**：本文公式分三等——报告原式（注明出处）、统一符号重写（结构不变，符号按本文约定）、教学示意式（只服务第一阶段理解与复现）。三者不混用；第 6 节的打分式是报告式 (1) 的统一符号重写（报告中 Gate 记为 $w^I_{t,j}$，本文记为 $g_{t,j}$）。

---

### 0. 先给结论：DSA 到底在做什么？

DeepSeek Sparse Attention（DSA）的核心不是“完全不看长上下文”，而是：

> **先用便宜的 Lightning Indexer 扫描全部历史 token，找出 Top-K；再让昂贵的 MLA Attention 只处理这些 Top-K 位置。**

可以把它理解成搜索系统里的：

> **粗排 → 精排**

![DeepSeek DSA 两阶段：轻量 Indexer 扫描全部历史，Sparse MLA 读取 Top-K](/images/deepseek-dsa-lightning-indexer/01_overview.svg)

*图 1：DSA 的两阶段结构。Indexer 便宜但全量扫描，MLA 昂贵但只读 Top-2048；两套 Cache 一套负责找位置、一套负责真正计算，不能互相替代。*

Lightning Indexer 的目标不是直接产生 Attention 输出，而是回答：

> **“对于当前 token，我最应该去历史的哪些位置取信息？”**

---

## 1. 第一件最容易混淆的事：参数和动态值不是一回事

学习 Lightning Indexer 时，我最开始反复混淆：

- `W_K`
- `k_t`
- `W_G`
- `g_t`
- `W_Q`
- `q_t`

核心原因是：**大写 W 是模型参数，小写 q/k/g 是某个 token 运行时产生的动态值。**

### 1.1 同一层中，参数是共享的

为了讲清楚，本文统一用下面的教学符号：

| 符号 | 含义 |
|---|---|
| {% raw %}$W_Q^I${% endraw %} | 生成 Indexer Query 的参数 |
| {% raw %}$W_K^I${% endraw %} | 生成 Indexer Key 的参数 |
| {% raw %}$W_G^I${% endraw %} | 生成 Gate / Head Mixer 的参数 |
| {% raw %}$q_{t,j}^I${% endraw %} | 当前 token {% raw %}$t${% endraw %} 的第 {% raw %}$j${% endraw %} 个 Indexer Query |
| {% raw %}$k_t^I${% endraw %} | token {% raw %}$t${% endraw %} 的 Indexer Key |
| {% raw %}$g_{t,j}${% endraw %} | 当前 token {% raw %}$t${% endraw %} 对第 {% raw %}$j${% endraw %} 个 Indexer Head 的动态 Gate |
| {% raw %}$r_{t,s}${% endraw %} | 当前 token {% raw %}$t${% endraw %} 对历史 token {% raw %}$s${% endraw %} 的 Indexer ranking score |

> 注：DeepSeek 论文/代码里 Gate 系数常写成 {% raw %}$w^I_{t,j}${% endraw %}。为了避免和参数矩阵 `W` 混淆，本文统一改写成 {% raw %}$g_{t,j}${% endraw %}。数学含义不变。

同一层里的所有 token 使用**同一套参数**：

```text
h1  ─┐
h2  ─┼──> 同一个 W_K^I ──> k1, k2, ...
h3  ─┤
...  │
ht  ─┘
```

所以：

{% raw %}
$$
k_t^I = W_K^I h_t
$$
{% endraw %}

`W_K^I` 共享，但 `h_t` 不同，因此每个 token 的 `k_t^I` 当然不同。

这和一个最简单的函数完全一样：

{% raw %}
$$
y=2x
$$
{% endraw %}

`2` 没有变，但输入 `x` 不同，输出 `y` 就不同。

---

### 1.2 “共享参数”不等于“输出相同”

下面这张表是整篇最值得记住的表：

| 对象 | 是模型参数吗 | 跨 token 共享吗 | 会随 token 改变吗 |
|---|---:|---:|---:|
| {% raw %}$W_Q^I${% endraw %} | ✅ | ✅ | ❌ |
| {% raw %}$W_K^I${% endraw %} | ✅ | ✅ | ❌ |
| {% raw %}$W_G^I${% endraw %} | ✅ | ✅ | ❌ |
| {% raw %}$q_t^I${% endraw %} | ❌ | ❌ | ✅ |
| {% raw %}$k_t^I${% endraw %} | ❌ | ❌ | ✅ |
| {% raw %}$g_t${% endraw %} | ❌ | ❌ | ✅ |

一句话：

> **共享的是“生成规则”，不是生成结果。**

---

## 2. 从当前 token 9 开始走一遍 Lightning Indexer

假设历史已经有：

```text
token1 token2 token3 ... token8
```

现在处理：

```text
token9
```

进入当前 Transformer 层之后，token9 已经有隐藏状态：

{% raw %}
$$
h_9 \in \mathbb{R}^{D}
$$
{% endraw %}

为了先理解原理，假设：

{% raw %}
$$
D=768
$$
{% endraw %}

Lightning Indexer 会围绕 `h9` 产生三类信息。

![DSA 机制图 1](/images/deepseek-dsa-lightning-indexer/05_paths.svg)

三者分工完全不同：

- `q9`：**我要从哪些检索角度找历史信息？**
- `g9`：**对于当前 token，这些检索角度各有多重要？**
- `k9`：**把 token9 自己编码成未来其他 token 可以查询的检索 Key。**

---

## 3. Indexer Query：当前 token “拿什么去搜”

假设 Indexer 有 {% raw %}$H_I${% endraw %} 个 Query Head，每个 Head 维度为 {% raw %}$d_I${% endraw %}。

教学版可以写成：

{% raw %}
$$
Q_t^I = W_Q^I h_t
$$
{% endraw %}

然后 reshape：

{% raw %}
$$
Q_t^I \in \mathbb{R}^{H_I \times d_I}
$$
{% endraw %}

例如：

{% raw %}
$$
H_I=4,\qquad d_I=64
$$
{% endraw %}

那么当前 token9 得到：

```text
q9,1 : 64 维
q9,2 : 64 维
q9,3 : 64 维
q9,4 : 64 维
```

多个 Query Head 可以理解成：

> **同一个当前 token，从多个不同的检索子空间去判断历史 token 是否相关。**

这里不要人为规定“Head1 一定负责实体、Head2 一定负责位置”。这些分工是训练自己学出来的。

---

## 4. Indexer Key：历史 token “被怎么检索”

每个 token 都会经过同一个 `W_K^I`：

{% raw %}
$$
k_s^I = W_K^I h_s
$$
{% endraw %}

于是历史缓存中有：

```text
token1 -> k1
token2 -> k2
token3 -> k3
...
token8 -> k8
```

这里 Lightning Indexer 有一个非常重要的设计：

> **Query 是多头的，但每个历史 token 只有一个共享 Indexer Key。**

也就是说判断 token5 时：

![DSA 机制图 2](/images/deepseek-dsa-lightning-indexer/06_shared_key.svg)

不是：

```text
q1 -> k_head1
q2 -> k_head2
q3 -> k_head3
```

而是多个 Query Head 都和同一个 `k5` 比较。

这样做的直觉是：

> **多 Query 保留多视角检索能力；单共享 Key 降低 Key Cache 和历史扫描成本。**

---

## 5. Gate：为什么必须是“当前 token 的 g_t”

这是我学习过程中最关键的一个理解点。

当前 token 会产生：

{% raw %}
$$
g_t = W_G^I h_t
$$
{% endraw %}

如果 Indexer 有 4 个 Query Head：

{% raw %}
$$
g_t=[g_{t,1},g_{t,2},g_{t,3},g_{t,4}]
$$
{% endraw %}

例如 token9：

{% raw %}
$$
g_9=[0.8,\ 0.2,\ -0.3,\ 1.1]
$$
{% endraw %}

这里的 `g9` **不是共享参数**。

共享的是：

{% raw %}
$$
W_G^I
$$
{% endraw %}

不同 token 经过同一个 {% raw %}$W_G^I${% endraw %}，会产生不同的 Gate：

```text
h8  -> W_G^I -> g8
h9  -> W_G^I -> g9
h10 -> W_G^I -> g10
```

因此：

{% raw %}
$$
g_8 \neq g_9 \neq g_{10}
$$
{% endraw %}

---

### 5.1 Gate 的真正含义

我最后把它理解为：

> **`g_t` 本质上是一个 query-conditioned head mixer。**

也就是：

> **站在当前 token 的角度，动态决定多个 Indexer Query Head 的判断应该怎么混合。**

这就是为什么必须使用**当前 token 的 `g_t`**。

当前的问题是：

> “token9 现在最需要从历史中找什么？”

而不是：

> “历史 token5 自己认为哪个 Head 比较重要？”

所以整个检索必须由当前 Query 驱动。

---

## 6. Lightning Indexer 怎么给历史 token 打分？

现在评价：

> token9 是否应该关注历史 token5？

token9 有多个 Query：

{% raw %}
$$
q_{9,1},q_{9,2},q_{9,3}
$$
{% endraw %}

token5 有一个：

{% raw %}
$$
k_5
$$
{% endraw %}

先分别做点积：

{% raw %}
$$
q_{9,1}^{T}k_5
$$
{% endraw %}

{% raw %}
$$
q_{9,2}^{T}k_5
$$
{% endraw %}

{% raw %}
$$
q_{9,3}^{T}k_5
$$
{% endraw %}

假设得到：

```text
Head1 =  2.0
Head2 = -1.0
Head3 =  3.0
```

先过 ReLU：

```text
Head1 = 2.0
Head2 = 0
Head3 = 3.0
```

当前 token9 的 Gate：

{% raw %}
$$
g_9=[0.8,\ 0.2,\ -0.3]
$$
{% endraw %}

于是：

```text
Head1: 2.0 ×  0.8 =  1.60
Head2: 0.0 ×  0.2 =  0.00
Head3: 3.0 × -0.3 = -0.90
```

最后加起来：

{% raw %}
$$
r_{9,5}=1.6+0-0.9=0.7
$$
{% endraw %}

这个：

{% raw %}
$$
\boxed{r_{9,5}=0.7}
$$
{% endraw %}

就是：

> **当前 token9 对历史 token5 的 Lightning Indexer ranking score。**

![Lightning Indexer 三个 Head 的 ReLU、Gate 与求和数值示意](/images/deepseek-dsa-lightning-indexer/02_scoring.svg)

*图 2：第 6 节数值例子的完整流水线。三个 Head 共享同一个 $k_5$，差别只来自各自的 Query 与当前 token 的 Gate；数值人为设定。*

---

### 6.1 它不是概率

Indexer Score：

- 可以大于 1；
- 可以小于 0；
- 没有固定的 {% raw %}$[0,1]${% endraw %} 或 {% raw %}$[-1,1]${% endraw %} 范围。

原因是：

{% raw %}
$$
\mathrm{ReLU}(q^Tk)\ge 0
$$
{% endraw %}

但 Gate：

{% raw %}
$$
g_{t,j}
$$
{% endraw %}

可以是正，也可以是负。

所以：

{% raw %}
$$
r_{t,s}
=
\sum_j
g_{t,j}\mathrm{ReLU}(q_{t,j}^Tk_s)
$$
{% endraw %}

理论上可以取正值或负值。

它的意义只是：

> **Ranking Score——用于比较历史 token 谁更值得被选。**

---

## 7. Top-K：排的是历史 token，不是 Head

这也是一个很容易误解的点。

Gate 并不是：

> 对 Head 排序，然后只保留几个 Head。

错误理解：

```text
Head1
Head2 -> Top-K Heads   ×
Head3
```

真正流程是：

```text
多个 Head
    ↓
共同给 token1 一个 score
共同给 token2 一个 score
共同给 token3 一个 score
...
    ↓
对“历史 token score”排序
```

例如：

| 历史 token | Ranking Score |
|---|---:|
| token1 | 0.3 |
| token2 | 2.8 |
| token3 | -0.4 |
| token4 | 1.1 |
| token5 | 0.7 |
| token6 | 3.7 |
| token7 | 2.1 |
| token8 | 0.2 |

如果：

{% raw %}
$$
K=3
$$
{% endraw %}

那么：

{% raw %}
$$
\text{TopK}=[6,2,7]
$$
{% endraw %}

Lightning Indexer 到这里就完成任务了。

它最终需要提供的核心结果就是：

> **Top-K 历史位置。**

---

## 8. 一个非常重要的问题：当前的 k9 当下不是没用吗？

最开始我有一个困惑：

> token9 当前查询历史时已经有 `q9 + g9` 了，那为什么还要算 `k9`？

答案有两层。

### 8.1 当前 causal self-attention 可以包含自己

官方 reference inference 的顺序是：

1. 计算 `k9`
2. 将 `k9` 写入 Indexer K Cache
3. `q9` 扫描到 `end_pos`

因此当前位置本身也可以参与 causal self-attention。

### 8.2 更重要：k9 是给未来 token 用的

即便为了教学把当前位置自己排除，`k9` 仍然必须计算。

因为 token10 来的时候需要：

```text
q10 + g10
    ↓
扫描
k1 k2 ... k8 k9 k10
```

因此：

```text
q9 : 当前查询使用，用完即可丢
g9 : 当前混合 Head 使用，用完即可丢
k9 : 写入 Indexer Cache，未来继续使用
```

---

## 9. 会不会缓存历史 h，然后每次重新算 k？

不会。

这是另一个关键误区。

如果系统每次都：

```text
保存 h1...h100000
然后每个 decode step：
h1 -> WK -> k1
h2 -> WK -> k2
...
```

那开销会非常离谱。

真实逻辑是：

![DSA 机制图 3](/images/deepseek-dsa-lightning-indexer/07_lifecycle.svg)

因此历史只需要缓存已经计算好的：

{% raw %}
$$
k_1,k_2,\ldots,k_t
$$
{% endraw %}

不需要为了 Indexer 缓存历史 `h`。

---

## 10. 每一步都重新计算 q_t 和 g_t，会不会很贵？

每生成一个新 token，只需要计算**当前 token 自己的**：

{% raw %}
$$
q_t,\quad g_t,\quad k_t
$$
{% endraw %}

不会重新计算历史：

{% raw %}
$$
q_1,\ldots,q_{t-1}
$$
{% endraw %}

也不会重新计算历史：

{% raw %}
$$
g_1,\ldots,g_{t-1}
$$
{% endraw %}

这些投影的成本本身与历史长度 {% raw %}$L${% endraw %} 无关。

真正随着上下文变长的成本来自：

> **当前 `q_t + g_t` 扫描历史 Indexer Key Cache。**

也就是说 DSA 没有让“全历史扫描”消失。

它只是把：

> **昂贵的主 Attention 全历史扫描**

替换成：

> **轻量 Indexer 全历史扫描 + Top-K 主 Attention**

---

## 11. Lightning Indexer 之后：进入 MLA 第二阶段

这里必须把两套 Cache 分开。

![DSA 机制图 4](/images/deepseek-dsa-lightning-indexer/08_dual_cache.svg)

### 第一套 Cache：Indexer Cache

主要保存：

{% raw %}
$$
k_s^I
$$
{% endraw %}

作用：

> **找历史位置。**

### 第二套 Cache：MLA 压缩缓存

保存 MLA 的压缩表示，例如：

{% raw %}
$$
c_s^{KV}
$$
{% endraw %}

以及位置编码相关缓存。

作用：

> **真正进行 Attention。**

这里本文默认讨论的是 MLA 压缩缓存，不把它重新解释为“缓存完整 K/V”。

---

## 12. Top-K 后发生什么？

假设 Lightning Indexer 得到：

{% raw %}
$$
[6,2,7]
$$
{% endraw %}

第二阶段根据这些位置访问 MLA 的压缩缓存：

```text
position 6 -> cKV6 + PE6
position 2 -> cKV2 + PE2
position 7 -> cKV7 + PE7
```

然后进行真正的 Sparse MLA Attention。

所以完整链路：

```text
当前 h_t
   ↓
当前 Indexer q_t + g_t
   ↓
扫描历史 Indexer k cache
   ↓
Ranking Scores
   ↓
Top-K indices
   ↓
按位置访问 MLA 压缩缓存
   ↓
Sparse MLA Attention
   ↓
o_t
```

这才是 DSA + MLA 在推理时最应该形成的脑图。

---

## 13. 为什么 DSA 要维护两套 Cache？

如果直接用 MLA 的主缓存做第一次“找位置”，Indexer 本身就会变重。

所以 DeepSeek 单独设计了一套非常轻量的 Indexer Key：

> **小 Key 负责检索，大信息留给 MLA。**

类比搜索系统：

| 阶段 | 类比 | 目标 |
|---|---|---|
| Lightning Indexer | 倒排/粗排 | 快速找到候选 |
| Top-K | 候选集合 | 缩小搜索空间 |
| MLA Attention | 精排/读取正文 | 高精度信息融合 |

![MHA、MLA 和 Indexer 每历史 token 每层核心缓存字节数对比](/images/deepseek-dsa-lightning-indexer/03_caches.svg)

*图 3：每历史 token、每层的缓存字节数（对数轴）。口径：MHA 基线为 K/V 各 128 头 × 128 维、BF16；MLA 为 $c^{KV}$ 512 维 + RoPE 64 维、BF16；Indexer Key 为 128 维、FP8，均不含量化 scale 等元数据。Indexer 的单位置成本比 MLA 压缩缓存还低近一个数量级——全量扫描在访存上才成立。*

---

## 14. 参数量：先看一个教学版

为了容易复现，设：

{% raw %}
$$
D=768,\qquad H_I=4,\qquad d_I=64
$$
{% endraw %}

教学版直接使用：

{% raw %}
$$
h_t\rightarrow W_Q^I\rightarrow H_I\times d_I
$$
{% endraw %}

忽略 bias 和 norm。

### Query 投影

{% raw %}
$$
768\times(4\times64)=196608
$$
{% endraw %}

### Key 投影

因为 Key 是共享单头：

{% raw %}
$$
768\times64=49152
$$
{% endraw %}

### Gate 投影

{% raw %}
$$
768\times4=3072
$$
{% endraw %}

### 总参数量

{% raw %}
$$
196608+49152+3072=248832
$$
{% endraw %}

即：

{% raw %}
$$
\boxed{\approx0.249M}
$$
{% endraw %}

对于教学模型而言，这个 Indexer 很小。

---

## 15. DeepSeek-V3.2-Exp 官方 Indexer 参数量

官方 671B 配置中：

```text
dim              = 7168
q_lora_rank      = 1536
index_n_heads    = 64
index_head_dim   = 128
index_topk       = 2048
```

需要注意：

> **官方 Query 路径并不是教学版的 `h -> WQ`。**

它复用了 MLA Query 的低秩中间表示：

```text
h_t
 ↓
MLA wq_a
 ↓
q_norm
 ↓
qr (1536维)
 ↓
Indexer wq_b
 ↓
64 × 128 个 Indexer Query 分量
```

因此 Indexer 自己新增的 Query 投影是：

{% raw %}
$$
W_{q,b}^{I}:1536\rightarrow64\times128
$$
{% endraw %}

#### 15.1 `wq_b`

{% raw %}
$$
1536\times(64\times128)
$$
{% endraw %}

{% raw %}
$$
=12,582,912
$$
{% endraw %}

约：

{% raw %}
$$
12.58M
$$
{% endraw %}

#### 15.2 `wk`

{% raw %}
$$
7168\times128=917,504
$$
{% endraw %}

约：

{% raw %}
$$
0.918M
$$
{% endraw %}

#### 15.3 `weights_proj`（本文 WG）

{% raw %}
$$
7168\times64=458,752
$$
{% endraw %}

约：

{% raw %}
$$
0.459M
$$
{% endraw %}

#### 15.4 `k_norm`

官方 reference `LayerNorm(128)` 有 weight + bias：

{% raw %}
$$
128+128=256
$$
{% endraw %}

#### 15.5 单层 Indexer 新增参数

约：

{% raw %}
$$
12,582,912+917,504+458,752+256
$$
{% endraw %}

{% raw %}
$$
\boxed{13,959,424\approx13.96M}
$$
{% endraw %}

如果简单按 61 层都配置 Indexer：

{% raw %}
$$
13.959M\times61\approx851.5M
$$
{% endraw %}

也就是约 **0.85B 参数**。

相对于 671B 总参数规模，它仍然只是较小的结构增量；更重要的问题其实不是参数量，而是**长上下文下 Indexer 的全历史扫描成本和访存模式**。

---

## 16. Indexer Cache 到底有多大？

官方：

{% raw %}
$$
d_I=128
$$
{% endraw %}

Indexer Key Cache 使用 FP8。

因此每个历史 token、每层的核心 Key 数据大约：

{% raw %}
$$
128\times1\text{ byte}=128\text{ bytes}
$$
{% endraw %}

不算量化 scale 等额外元数据。

以 128K 上下文为例：

{% raw %}
$$
131072\times128
=
16,777,216\text{ bytes}
$$
{% endraw %}

即核心 Key 数据约：

{% raw %}
$$
\boxed{16\text{ MiB / layer}}
$$
{% endraw %}

这说明 Lightning Indexer 之所以能做全历史扫描，一个基础前提就是：

> **每个历史位置的检索表示必须足够小。**

---

## 17. 它真的省计算吗？

这里不能简单说：

> 128K → 2048，所以就是 64 倍加速。

这是错误的。

因为 Lightning Indexer 自己仍然要扫 128K。

---

### 17.1 Indexer 的粗略扫描量

官方配置：

{% raw %}
$$
H_I=64,\qquad d_I=128
$$
{% endraw %}

对于单个新 Query、128K 历史：

{% raw %}
$$
L\times H_I\times d_I
$$
{% endraw %}

约为：

{% raw %}
$$
131072\times64\times128
$$
{% endraw %}

{% raw %}
$$
\approx1.07\times10^9
$$
{% endraw %}

即约 **1.07B 量级的乘加匹配工作**。

所以 Lightning Indexer 自己绝对不是“免费”的。

---

### 17.2 但主 Attention 被压到了 Top-2048

如果主 MLA 每次都对 128K 历史做真正的高维 Attention，成本明显更高。

DSA 的关键变成：

```text
便宜但全量：
Indexer × 128K

+

昂贵但稀疏：
MLA × 2048
```

而不是：

```text
昂贵：
MLA × 128K
```

以官方维度做一个**只用于量级理解**的简化估算：

- Indexer 扫描约：1.07B
- Top-2048 MLA score 部分约：0.15B
- Dense 128K MLA score 部分约：9.66B

简化比值：

{% raw %}
$$
\frac{9.66}{1.07+0.15}\approx7.9
$$
{% endraw %}

即这个极简计算模型下，Attention score 相关交互大约是 **7.9× 的理论量级差异**。

但这**绝不是端到端真实加速比**，因为真实运行还包括：

- projection；
- Top-K reduction；
- Kernel 启动；
- HBM 访问；
- 稀疏 gather；
- 多卡通信；
- FP8 量化/反量化；
- MoE 等其他模块。

真正性能必须依赖专门的 Indexer Kernel 和 Sparse Attention Kernel。

![Dense MLA 与 DSA 简化打分乘加量随上下文长度变化](/images/deepseek-dsa-lightning-indexer/04_compute.svg)

*图 4：本文简化模型下，单步解码的打分乘加量随上下文长度的变化。Dense 按 MLA 的 128 头 × 576 维对全部 L 打分；DSA 为 Indexer（64 头 × 128 维）扫 L 加 MLA 对 Top-2048 打分。只比较打分环节，不含投影、gather、通信与 MoE，不是端到端延迟。*

---

## 18. 补上训练视角：Indexer 是怎么学会“该看谁”的

前面所有内容都在讲推理：给定一个训练好的 Indexer，它如何打分、如何选位置。但有一个问题不能跳过：

> **Indexer 的打分凭什么可信？选错了怎么办？**

DeepSeek 的答案不是让 Indexer 从语言建模 loss 里顺带学会选位置，而是给它一个**单独的对齐目标**：模仿主 Attention 自己的分布。[6, §2.1]

### 18.1 目标分布：主 Attention 说“重要”的样子

对第 $t$ 个 query token，把主 Attention 在全部注意力头上的分数求和，再沿序列维做 L1 归一化，得到目标分布 $p_{t,:}$。Indexer 分数经 Softmax 后与它对齐：

{% raw %}
$$
L_I = D_{KL}\big(\mathrm{Softmax}(I_{t,:})\ \|\ p_{t,:}\big)
$$
{% endraw %}
> 统一符号重写：结构与报告式 (3) 一致，$I_{t,:}$ 即本文的 $r_{t,:}$。

注意这个目标的真实含义：**Indexer 学的不是"语义相关"，而是"主 Attention 实际会把权重放在哪"**。它是一个为稠密注意力打工的代理学生。

### 18.2 两阶段：先稠密热身，再稀疏训练

**Dense Warmup 阶段**：保持稠密注意力，冻结主模型全部参数，只训 Indexer，让它先对齐稠密分布。学习率 $10^{-3}$，1000 步，每步 16 条 128K 序列，共约 2.1B token。

**Sparse Training 阶段**：开启 Top-K 选择（每个 query 选 2048 个 KV），主模型与 Indexer 一起训练，但目标分开：

- 主模型只用语言建模 loss；
- Indexer 只用 KL loss，且只在被选中的集合 $S_t$ 上计算（报告式 (4)）；
- Indexer 的输入从计算图上 detach，避免两个目标互相拉扯。

学习率 $7.3\times10^{-6}$，15000 步，每步 480 条 128K 序列，共约 943.7B token。报告口径下，这是基于 V3.1-Terminus 的继续训练，不是 DSA 从零训练的成本，引用这个数字时不要错位。

### 18.3 创新的边界

每个创新都要讲它的代价。DSA 的代价不在推理链路里，在下面四处：

1. **选错不可恢复**：某一层里没被选进 Top-K 的位置，该层主 Attention 就永远读不到它。Indexer 的召回率就是 DSA 的效果上限——这正是训练目标必须选"对齐稠密分布"、而不是某个更省事的代理指标的原因。
2. **继承稠密注意力的偏置**：Indexer 是稠密注意力的学生。稠密注意力系统性忽略的位置，Indexer 也学不会选。它优化的是保真度，不是纠错。
3. **固定 K 与内容无关**：2048 对所有 query 一视同仁。信息高度分散的长文档可能不够用，简单上下文又纯属浪费；K 是部署前定死的超参，不是按内容自适应的预算。
4. **理论降幅不等于延迟降幅**：第 17 节的 7.9× 只是打分环节的简化账。真实收益依赖 DeepGEMM 的 Indexer logit kernel 与 FlashMLA 的稀疏 kernel；没有这套 kernel，gather 与 Top-K reduction 的开销会吃掉相当一部分理论收益。

---

## 19. 我学习过程中真正踩过的误区

这一节比公式更值得以后复习。

### 误区 1：`W_K` 共享，所以 `k_t` 也应该共享

错误。

正确关系：

{% raw %}
$$
k_t=W_Kh_t
$$
{% endraw %}

`W_K` 是共享机器。

`k_t` 是不同 token 的不同输出。

---

### 误区 2：Head weights 是一组固定参数

不是。

共享的是生成 Gate 的参数：

{% raw %}
$$
W_G
$$
{% endraw %}

每个当前 token 都动态产生：

{% raw %}
$$
g_t=W_Gh_t
$$
{% endraw %}

所以：

{% raw %}
$$
g_8\neq g_9\neq g_{10}
$$
{% endraw %}

---

### 误区 3：Gate 是用来 Top-K Head

不是。

Gate 是：

> **query-conditioned head mixer**

它负责把多个 Query Head 对某个历史 token 的匹配结果混合起来。

真正 Top-K 排序的是：

{% raw %}
$$
r_{t,1},r_{t,2},...,r_{t,L}
$$
{% endraw %}

也就是**历史 token 的 ranking score**。

---

### 误区 4：为什么用当前 g_t，而不是历史 g_s？

因为检索问题本身是：

> **当前 token 想看谁？**

所以需要站在当前 Query 的角度决定：

> 当前这一次查询，各个 Indexer Head 应该有多重要。

因此 `g_t` 必须是当前 token 的。

---

### 误区 5：每次是不是要重新计算所有历史 k？

不是。

历史：

{% raw %}
$$
k_1,k_2,\ldots,k_{t-1}
$$
{% endraw %}

已经缓存。

当前只计算：

{% raw %}
$$
k_t
$$
{% endraw %}

然后 append 到 Indexer K Cache。

---

### 误区 6：历史 h 也要一直缓存

至少对 Lightning Indexer 本身而言，不需要为了重新生成 k 而缓存历史 h。

历史 k 已经生成并缓存。

---

### 误区 7：当前 k_t 没用，所以不用算

不对。

它：

1. 当前 causal self-attention 中可以被当前位置使用；
2. 更重要的是要留下给未来 token 查询。

因此：

```text
q_t -> 当前用，之后丢
g_t -> 当前用，之后丢
k_t -> 当前生成，并进入历史 cache
```

---

### 误区 8：Top-K 后还要恢复历史完整 K/V

在 DSA + MLA 的讨论中，更准确的理解是：

> Top-K 提供位置，然后主 Attention 访问这些位置的 **MLA 压缩 latent cache（如 cKV + PE）**。

不要把思路重新退化成必须先保存/恢复完整传统 K/V。

---

## 20. 最终完整推理链路

如果只能保存一张图，就保存下面这张。

![DSA 机制图 5](/images/deepseek-dsa-lightning-indexer/09_full_flow.svg)

可以浓缩成：

{% raw %}
$$
\boxed{
h_t
\rightarrow
(q_t,g_t,k_t)
\rightarrow
\text{scan cached }k
\rightarrow
TopK
\rightarrow
\text{select MLA latent cache}
\rightarrow
Sparse\ MLA
\rightarrow
o_t
}
$$
{% endraw %}

---

## 21. 一句话理解 DSA、MLA 和 Lightning Indexer 的分工

### MLA

解决：

> **每个历史 token 要缓存多少 Attention 信息？**

核心思想：

{% raw %}
$$
K/V\rightarrow compressed\ latent
$$
{% endraw %}

---

### Lightning Indexer

解决：

> **当前 token 到底值得读取哪些历史位置？**

核心：

{% raw %}
$$
q_t+g_t
\rightarrow
\text{scan index keys}
\rightarrow
TopK
$$
{% endraw %}

---

### DSA

把两者组合：

> **全历史做便宜检索，Top-K 做昂贵 Attention。**

---

## 22. 下一步复现时，我应该先写什么？

不建议一开始就完整复刻 DeepSeek 的 FP8、RoPE、MLA 和 CUDA Sparse Kernel。

应该分三阶段。

### Stage 1：纯教学版 Lightning Indexer

先实现：

```python
q = Wq(h_current)
g = Wg(h_current)
k_new = Wk(h_current)

k_cache.append(k_new)

match = relu(q @ k_cache.T)
score = (match * g).sum(head_dim)

indices = topk(score, K)
```

验收：

- 能明确区分参数和 activation；
- 历史只缓存 Indexer Key；
- 当前 q/g 不进 cache；
- Top-K 排的是历史 token。

---

### Stage 2：加入 MLA 压缩缓存

维护：

```text
index_k_cache
mla_c_kv_cache
mla_pe_cache
```

流程：

```python
topk_indices = indexer(...)

selected_c_kv = c_kv_cache[topk_indices]
selected_pe = pe_cache[topk_indices]

out = sparse_mla_attention(
    current_query,
    selected_c_kv,
    selected_pe
)
```

这一步完成之后，DSA + MLA 的数据流就真正跑通了。

---

### Stage 3：再处理工程优化

最后再考虑：

- FP8 Indexer Key；
- RoPE；
- block/paged cache；
- streaming Top-K；
- Sparse Attention Kernel；
- DeepGEMM / FlashMLA；
- 与官方权重兼容。

不要反过来。

---

## 23. 最后的自测

如果下面 8 个问题都能直接回答，说明 Lightning Indexer 推理原理已经真正理解。

#### Q1

`W_K` 和 `k_t` 哪个共享？

> `W_K` 是同一层跨 token 共享的训练参数；`k_t` 是 token t 的动态 Key，不共享。

#### Q2

为什么 `q_t` 不缓存？

> 它只服务当前 Query，未来 token 会生成自己的 Query。

#### Q3

为什么 `k_t` 要缓存？

> 它会成为未来 token 查询的历史 Indexer Key。

#### Q4

为什么每一步都要重新算 `g_t`？

> 因为 `g_t` 是 query-conditioned head mixer，必须站在当前 token 的检索需求决定各 Indexer Head 如何混合。

#### Q5

Top-K 排的是 Head 吗？

> 不是。排的是各历史 token 的最终 ranking score。

#### Q6

历史 hidden state 是否需要为了 Indexer 一直缓存？

> 不需要。已经转换后的历史 Indexer Key 被缓存即可。

#### Q7

Top-K 后访问什么？

> 对应历史位置的 MLA 压缩缓存，如 `cKV + PE`。

#### Q8

DSA 为什么仍然有意义，明明 Indexer 还是扫描整个历史？

> 因为全历史阶段使用小型、低精度、共享 Key 的廉价 Indexer；昂贵的 MLA 主 Attention 只处理 Top-K。

如果这 8 个问题能自然说出来，Lightning Indexer 的推理主线就已经过关。

---

## 24. 官方实现与本文符号对照

| 本文 | DeepSeek-V3.2-Exp reference code |
|---|---|
| {% raw %}$W_Q^I${% endraw %} | `Indexer.wq_b` |
| {% raw %}$W_K^I${% endraw %} | `Indexer.wk` |
| {% raw %}$W_G^I${% endraw %} | `Indexer.weights_proj` |
| {% raw %}$k${% endraw %} normalization | `Indexer.k_norm` |
| 当前 Gate {% raw %}$g_t${% endraw %} | `weights_proj(x) * n_heads**-0.5` |
| Indexer cache | `k_cache` + `k_scale_cache` |
| Top-K | `index_score.topk(...)` |
| MLA latent cache | `kv_cache` |
| MLA positional cache | `pe_cache` |

官方 Query 路径还会复用 MLA 的：

```text
x
 -> wq_a
 -> q_norm
 -> qr
 -> Indexer.wq_b
```

所以本文的：

{% raw %}
$$
q=W_Qh
$$
{% endraw %}

只是用于第一阶段理解和复现的简化写法。

---

## 参考资料

1. [DeepSeek-V3.2-Exp 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

2. [DeepSeek-V3.2-Exp 官方 reference inference：`inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py)

3. [DeepSeek-V3.2-Exp 671B 配置](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json)

4. [Hugging Face DeepSeek-V3.2 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v32/modular_deepseek_v32.py)

5. DeepSeek 官方仓库说明中还给出了高性能 Indexer logit kernel（DeepGEMM）与 Sparse Attention kernel（FlashMLA）的入口，可用于后续工程复现。

6. [DeepSeek-V3.2 技术报告（arXiv:2512.02556），§2.1 DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556)

---

## 核验范围

- **已核对**：index score 的形式与 ReLU + Gate 结构（报告 §2.1 式 (1)–(2)；报告中 Gate 记为 $w^I_{t,j}$，本文改记 $g_{t,j}$）；Indexer 两阶段训练与 KL 目标（报告 §2.1 式 (3)–(4)：warmup 1000 步、学习率 $10^{-3}$、约 2.1B token；稀疏阶段 15000 步、学习率 $7.3\times10^{-6}$、约 943.7B token、Top-K = 2048、Indexer 输入 detach、两目标分开优化）。
- **已核对**：671B 配置（dim 7168、q_lora_rank 1536、index_n_heads 64、index_head_dim 128、index_topk 2048）出自官方仓库 `config_671B_v3.2.json`；第 15 节参数量为本文按配置逐项乘算的结果，不是报告原文数字。
- **本文估算**：第 17 节与图 4 的计算量只含打分环节，是简化模型，不是实测数据；图 3 的缓存口径见图注。
- **未确认**：Indexer 的 q/k 上 RoPE 的具体施加细节（报告图 2 有示意）与 `k_scale_cache` 的存储格式，本文未展开，不影响推理主线结论。

---

## 最终记忆卡

```text
【共享参数】
W_Q^I / W_K^I / W_G^I
        │
        ▼
【当前 token 动态值】
q_t / k_t / g_t
        │
        ├── k_t 写入 Indexer Cache
        │
        └── q_t + g_t 扫历史 k
                     │
                     ▼
              Ranking Scores
                     │
                     ▼
                   Top-K
                     │
                     ▼
         读取 MLA 压缩 cKV + PE
                     │
                     ▼
             Sparse MLA Attention
                     │
                     ▼
                    o_t
```

> **最重要的一句话：**
>
> Lightning Indexer 使用当前 token 的 `q_t` 与 `g_t`，去评价历史缓存中的 Indexer Key；其中 `g_t` 是一个 **query-conditioned head mixer**。Top-K 只负责给出历史位置，真正的信息读取仍由 MLA 在这些位置的压缩缓存上完成。
