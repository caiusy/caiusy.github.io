---
title: 结合 MiniMind 源码彻底搞懂 LoRA 与 MoE
date: 2026-08-30 20:00:00
categories:
  - 多模态 AI
  - 大模型
tags:
  - LoRA
  - MoE
  - Transformer
  - MiniMind
type: article
difficulty: intermediate
mathjax: true
description: >
  从直觉到公式，从公式到 Tensor Shape，从 Shape 到 MiniMind 真实代码——
  彻底搞懂 LoRA 低秩适配与 MoE 混合专家的工作原理、工程实现和设计取舍。
review_status: published
---

大模型里经常同时出现两个缩写：LoRA 和 MoE。它们看起来都在"优化模型"，但优化的方向完全不同。

**LoRA** 关心的是：模型已经很大了，怎样用极少的可训练参数完成微调？
**MoE** 关心的是：怎样让模型拥有更大的参数容量，但每个 token 不必经过所有参数？

这篇文章不从概念图开始，而是沿着 `vendor/minimind-master` 中的真实代码，把一次 token 的数据流走一遍——同时补齐背后的数学推导和 Tensor Shape。

<!-- more -->

---

## 一、LoRA 的直觉：微调不需要全部权重参与

### 1.1 全参微调的代价

全参数微调（Full Fine-tuning）的显存代价公式：

{% raw %}
$$
\text{显存} \approx 4 \times |\theta| \text{ bytes（fp32 参数 + 梯度 + Adam 一/二阶矩）}
$$
{% endraw %}

对一个 7B 模型，约需 **112 GB 显存**，远超单卡 A100 的 80 GB 上限。

更关键的是一个实验发现（Aghajanyan et al., 2020）：**预训练权重在任务适配时的真实变化量具有极低的内在秩**。换句话说，微调时权重矩阵的更新量 {% raw %}$\Delta W${% endraw %} 虽看起来是全矩阵，但其有效信息量极少。

### 1.2 低秩分解：把 ΔW 拆成两个瘦矩阵

设原始权重矩阵 {% raw %}$W_0 \in \mathbb{R}^{d \times k}${% endraw %}，正常微调产生：

{% raw %}
$$
W = W_0 + \Delta W, \quad \Delta W \in \mathbb{R}^{d \times k}
$$
{% endraw %}

LoRA 的核心假设：{% raw %}$\text{rank}(\Delta W) = r \ll \min(d, k)${% endraw %}，因此分解为：

{% raw %}
$$
\Delta W = B A, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k}
$$
{% endraw %}

**参数量对比**（以 {% raw %}$d = k = 4096,\; r = 8${% endraw %} 为例）：

| 方案 | 参数量 | 压缩比 |
|:---|:---:|:---:|
| 全量 {% raw %}$\Delta W${% endraw %} | 16,777,216 | 1× |
| LoRA {% raw %}$B + A${% endraw %} | 65,536 | **256×** |

![LoRA 原始 Linear 与低秩旁路结构：x 同时经过冻结的 W₀ 和可训练的 B·A，输出相加](/images/lora-moe-deep-dive/lora-diagram.svg)

### 1.3 前向传播与 Tensor Shape

LoRA 的前向等价于：

{% raw %}
$$
h = W_0 x + \frac{\alpha}{r} \cdot B(Ax)
$$
{% endraw %}

Tensor Shape 全程追踪（{% raw %}$B${% endraw %} 为 batch，{% raw %}$T${% endraw %} 为序列长度）：

```text
输入 x              [B, T, k]
A 的输出 Ax         [B, T, r]   ← r=8，极小的瓶颈
B 的输出 B(Ax)      [B, T, d]
原路径 W₀x          [B, T, d]
缩放 (α/r)·B(Ax)   [B, T, d]
最终输出 h          [B, T, d]   ← 与原始输出同形
```

### 1.4 初始化策略：保证训练开始时 ΔW = 0

- **A** 用 Kaiming 均匀分布初始化（有方差，提供非零输入）；
- **B** 初始化为**全零**；
- 因此 {% raw %}$BA = 0${% endraw %}，不改变预训练模型的初始行为。

缩放因子 {% raw %}$\frac{\alpha}{r}${% endraw %} 解耦了秩 {% raw %}$r${% endraw %} 与梯度量级的关系——不管 {% raw %}$r${% endraw %} 取多大，学习率的影响都通过 {% raw %}$\alpha${% endraw %} 统一控制（常设 {% raw %}$\alpha = r${% endraw %} 或 {% raw %}$2r${% endraw %}）。

---

## 二、MiniMind 中 LoRA 的真实实现

### 2.1 LoRA 模块结构

MiniMind 用一个嵌套子模块实现 LoRA：

```python
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.linear = linear           # 原始冻结权重 W₀
        d, k = linear.weight.shape
        self.lora_A = nn.Linear(k, r, bias=False)   # A: [r, k]
        self.lora_B = nn.Linear(r, d, bias=False)   # B: [d, r]
        self.scale  = alpha / r

        # 初始化：A 用 Kaiming，B 全零
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # x: [B, T, k]
        base = self.linear(x)                       # [B, T, d]  冻结路径
        lora = self.lora_B(self.lora_A(x))          # [B, T, d]  低秩路径
        return base + self.scale * lora             # [B, T, d]
```

### 2.2 apply_lora()：哪些层挂 LoRA

```python
def apply_lora(model, r=8, alpha=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 只对方阵层挂 LoRA（Q、K、V、O proj）
            if module.weight.shape[0] == module.weight.shape[1]:
                parent, attr = get_parent(model, name)
                setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha))
```

> **注意**：当前实现只匹配方阵 Linear（输入维度 = 输出维度），因此 FFN 的 gate/up/down projection 不会自动加 LoRA。

### 2.3 梯度只流入 LoRA 参数

```python
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False   # 冻结所有非 LoRA 参数
```

反向传播链（{% raw %}$\mathcal{L}${% endraw %} 为 loss）：

{% raw %}
$$
\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} \cdot (Ax)^\top, \quad
\frac{\partial \mathcal{L}}{\partial A} = B^\top \frac{\partial \mathcal{L}}{\partial h} \cdot x^\top
$$
{% endraw %}

{% raw %}$W_0${% endraw %} 的 `requires_grad=False` 让 PyTorch 不为其积累梯度；AdamW 只维护 {% raw %}$A${% endraw %} 和 {% raw %}$B${% endraw %} 的一/二阶矩状态。

### 2.4 保存 / 加载 / 合并

**只保存 LoRA 参数**（checkpoint 极小）：

```python
lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
torch.save(lora_state, "lora_checkpoint.pt")
```

**推理时合并，零额外开销**：

```python
# W_merged = W₀ + (α/r) · B·A
module.linear.weight.data += (
    module.scale * module.lora_B.weight @ module.lora_A.weight
)
```

合并后的权重满足：

{% raw %}
$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$
{% endraw %}

合并后可以移除 LoRA 分支，推理计算图恢复成普通 Linear，**无任何额外推理延迟**。

### 2.5 显存节省的量化

以 MiniMind-0.1B（dim=512，8 层）为例，仅对 Q、V 挂 LoRA（{% raw %}$r=16${% endraw %}）：

```text
每层可训练参数:
  A_q: [16, 512] = 8,192
  B_q: [512, 16] = 8,192
  A_v: [16, 512] = 8,192
  B_v: [512, 16] = 8,192
  小计: 32,768 / 层

8 层合计: 32,768 × 8 = 262,144 ≈ 0.26 M

全参微调总显存:  ~1,600 MB
LoRA 微调总显存: ~204  MB   ← 节省约 8×
```

---

## 三、MoE：稀疏激活换取更大容量

### 3.1 为什么需要 MoE

增大稠密模型（Dense Model）有一条硬约束：**每个 token 都经过所有参数**。
若将 FFN 维度翻 4 倍，推理 FLOP 也翻 4 倍——线性增长。

MoE 打破这条约束：**增大参数量，但每个 token 只激活其中一个子集**。

| | Dense FFN | MoE（E=4, K=1）|
|---|:---:|:---:|
| 总参数量 | {% raw %}$2 d \cdot d_{ff}${% endraw %} | {% raw %}$4 \times 2 d \cdot d_{ff}${% endraw %} |
| 每 token 激活参数 | {% raw %}$2 d \cdot d_{ff}${% endraw %} | {% raw %}$1 \times 2 d \cdot d_{ff}${% endraw %} |
| 性价比 | 1× | **容量 4×，算力持平** |

### 3.2 Router 路由机制

Router 是 MoE 的决策核心，输入 token hidden state {% raw %}$h \in \mathbb{R}^d${% endraw %}：

{% raw %}
$$
G(h) = \text{Softmax}(W_g \cdot h), \quad W_g \in \mathbb{R}^{E \times d}
$$
{% endraw %}

Top-K 选择（MiniMind 默认 K=1）：

{% raw %}
$$
\text{TopK}(G(h),\, K) \quad \Rightarrow \quad \text{indices} \in \mathbb{Z}^K,\;\; \text{weights} \in \mathbb{R}^K
$$
{% endraw %}

Tensor Shape 追踪（{% raw %}$B=2, T=512, d=512, E=4, K=1${% endraw %}）：

```text
h               [B, T, d]      = [2, 512, 512]
展平            [B·T, d]       = [1024, 512]
W_g @ h.T       [B·T, E]       = [1024, 4]     路由 logit
Softmax         [B·T, E]       = [1024, 4]     路由概率
TopK(K=1)
  weights       [B·T, 1]       = [1024, 1]
  indices       [B·T, 1]       = [1024, 1]     选中的专家 ID
```

![MoE Router 将每个 token 动态分发到激活专家，其余专家休眠不参与计算](/images/lora-moe-deep-dive/moe-routing.svg)

---

## 四、MiniMind 中 MoE 的真实实现

### 4.1 普通 FFN（对照组）

MiniMind 的 FFN 是 SwiGLU 形式：

```python
def forward(self, x):
    return self.down_proj(
        self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    )
```

写成数学形式：

{% raw %}
$$
\operatorname{FFN}(x) = W_{\text{down}}\!\left(\operatorname{SiLU}(W_{\text{gate}}\,x) \odot W_{\text{up}}\,x\right)
$$
{% endraw %}

所有 token 都经过同一个 FFN。

### 4.2 MoE FFN

`MOEFeedForward` 初始化时创建多个独立的 FFN：

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate    = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        # 默认: num_experts=4, num_experts_per_tok=1
```

### 4.3 一次完整 MoE 前向

**Step 1：展平 + 打分**

```python
x_flat = x.view(-1, hidden_dim)                        # [B·T, d]
scores = F.softmax(self.gate(x_flat), dim=-1)          # [B·T, E]
```

某个 token 的分数示例：`[0.05, 0.10, 0.80, 0.05]`

**Step 2：Top-K 选择 + 归一化**

```python
topk_weight, topk_idx = torch.topk(
    scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
)
topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
```

Top-1 时，上面的 token 选中专家 2，权重 0.80（归一化后 = 1.0）。

**Step 3：分发 token，聚合结果**

```python
y = torch.zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask     = (topk_idx == i)               # [B·T, K] bool
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight    = topk_weight[mask].view(-1, 1)
        y.index_add_(
            0,
            token_idx,
            (expert(x_flat[token_idx]) * weight).to(y.dtype)
        )
```

假设 8 个 token 的路由结果：

```text
Expert 0 → token 1, 6
Expert 1 → token 0, 4, 7
Expert 2 → token 2, 3
Expert 3 → token 5
```

各专家只计算属于自己的 token，`index_add_` 写回原位置。

**Step 4：整合输出**

若 Top-K=2，两个专家的输出按权重加权求和：

{% raw %}
$$
y = w_1 E_1(x) + w_2 E_2(x), \quad w_1 + w_2 = 1
$$
{% endraw %}

### 4.4 一个容易忽略的工程细节

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

这段代码数值上等于加零，但把**未被当前 batch 选中的 Expert 参数接入了计算图**。
原因：分布式训练（DDP）中，若某专家在整个 batch 中没有任何 token，框架会把它标记为 `unused parameters`，影响梯度同步。这个写法保留图连接，不改变前向数值。

---

## 五、负载均衡：为什么必须有 auxiliary loss

### 5.1 路由崩塌问题

Router 容易出现塌缩——几乎所有 token 都选择同一个专家，其余专家成为"死亡专家"。

### 5.2 辅助均衡损失

设第 {% raw %}$i${% endraw %} 个专家的实际 token 比例为 {% raw %}$f_i${% endraw %}，Router 平均概率为 {% raw %}$P_i${% endraw %}：

{% raw %}
$$
\mathcal{L}_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i \cdot \lambda_{\text{aux}}
$$
{% endraw %}

MiniMind 的实现：

```python
load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # f_i
self.aux_loss = (
    load * scores.mean(0)      # f_i × P_i
).sum() * self.config.num_experts * self.config.router_aux_loss_coef
```

{% raw %}$f_i P_i${% endraw %} 同时惩罚：占用过多的专家（大 {% raw %}$f_i${% endraw %}）和路由概率过高的专家（大 {% raw %}$P_i${% endraw %}），双重压力迫使负载趋于均匀。

`router_aux_loss_coef` 默认为 `5e-4`——它是约束项，不应压过语言建模主损失：

{% raw %}
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}}
$$
{% endraw %}

---

## 六、LoRA 与 MoE 的组合

![LoRA 与 MoE 各自解决的问题与核心设计对比](/images/lora-moe-deep-dive/comparison.svg)

两者不互斥，但要区分两个层次：

```text
MoE  → 决定有哪些专家，以及 token 走哪个专家（预训练/架构层）
LoRA → 决定哪些权重用低秩方式适配新任务（微调/适配层）
```

组合后的典型结构：

```text
Transformer Block
├── Attention
│   ├── q_proj  ← 可挂 LoRA
│   ├── k_proj
│   ├── v_proj  ← 可挂 LoRA
│   └── o_proj  ← 可挂 LoRA
└── MoE FFN
    ├── Router  ← 可选：冻结 or 训练
    ├── Expert 0: gate/up/down proj  ← 可挂独立 LoRA
    ├── Expert 1: gate/up/down proj
    ├── Expert 2: gate/up/down proj
    └── Expert 3: gate/up/down proj
```

**Router 是否训练**是一个关键选择：

| | 冻结 Router | 训练 Router |
|---|---|---|
| 专家分工 | 保持预训练分工 | 可重新分配 |
| 稳定性 | 更稳定 | 更容易负载失衡 |
| 适用场景 | 领域迁移 | 新任务分布差异大 |

> **当前 MiniMind 的限制**：`apply_lora()` 只匹配方阵 Linear，Expert 的 gate/up/down projection（非方阵）不会自动加 LoRA。若要做专家 LoRA，需显式匹配 `experts.*.gate_proj` 等目标模块。

---

## 七、实用选择指南

```text
目标是微调格式 / 领域表达 / 工具调用
→ 优先 LoRA：改动小，checkpoint 小，训练成本低

目标是提升知识容量 / 多技能并存
→ 考虑 MoE：需从预训练阶段就设计专家结构

最优路线：
  dense 模型验证数据和任务
        ↓
  LoRA 快速验证微调收益
        ↓
  确认任务规模后再评估 MoE
```

> ⚠️ **特别注意**：`use_moe=1` 不是推理开关。MoE 权重结构不同，必须加载对应的 MoE checkpoint，与 dense 权重不能混用。

---

## 八、四个公式压缩全文

LoRA 旁路：

{% raw %}
$$
h = W_0 x + \frac{\alpha}{r} B A x
$$
{% endraw %}

低秩约束：

{% raw %}
$$
\operatorname{rank}(BA) \le r \ll \min(d, k)
$$
{% endraw %}

MoE 稀疏混合：

{% raw %}
$$
y = \sum_{i \in \operatorname{TopK}(x)} w_i\, E_i(x), \quad \sum w_i = 1
$$
{% endraw %}

MoE 训练目标：

{% raw %}
$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{aux}}
$$
{% endraw %}

---

**LoRA** 是在原模型旁边学习一个低秩补丁，**MoE** 是在 FFN 位置建立专家组并让 Router 动态分流。
一个负责少训练参数，一个负责多模型容量。

理解这两个差别，后面再看 QLoRA、Mixtral、专家并行或 MoE-LoRA，就不会只停留在名词层面了。

---

## 参考文献

1. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
2. Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017.
3. Jiang et al. (2024). *Mixtral of Experts*. arXiv:2401.04088.
4. Aghajanyan et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ACL 2021.
5. Fedus et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models*. JMLR 2022.
