---
title: MoE 深度解析：Switch Transformer 与 Mixtral 的稀疏之道
date: 2026-02-08 10:00:00
updated: 2026-02-08 10:00:00
tags:
  - 深度学习
  - MoE
  - Switch Transformer
  - Mixtral
  - 大模型架构
categories:
  - AI原理深究
mathjax: true
description: "【费曼式深度解析】从零理解 Mixture of Experts 的核心机制。详解 Router 路由、Load Balance Loss 推导、梯度流动分析，含完整 Tensor 维度变化图解与生产级代码实现。"
---

<style>
mjx-container { font-size: 0.7em !important; }
.MathJax { font-size: 0.7em !important; }
code { font-family: 'Fira Code', monospace; }
h2 { margin-top: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
.feynman-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }
.feynman-box h4 { color: #fff; margin-top: 0; }
.insight-box { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 15px; margin: 15px 0; }
.warning-box { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 15px 0; }
.quiz-box { background: #ecfdf5; border: 2px dashed #10b981; padding: 15px; margin: 20px 0; border-radius: 8px; }
</style>

# MoE 深度解析：Switch Transformer 与 Mixtral 的稀疏之道

> 📅 **创建时间**：2026-02-08
> 🏷️ **标签**：#MoE #SwitchTransformer #Mixtral #稀疏计算 #大模型
> 📚 **学习方法**：费曼式讲解 + 数学推导 + 代码实现
> 📖 **前置知识**：Transformer FFN 层, Softmax, PyTorch 基础

---

## 🎯 一句话理解 MoE

<div class="feynman-box">
<h4>🧠 费曼式理解</h4>
<p><strong>MoE 就像一家拥有 128 个专科医生的医院。</strong></p>
<p>普通医院：所有病人都找同一个全科医生（稠密 FFN）→ 医生累死，效率低下。</p>
<p>MoE 医院：前台护士（Router）快速判断病情，把心脏病人分给心脏专家，骨折病人分给骨科专家。每个病人只见 1-2 个专家，但医院总共有 128 个专家随时待命。</p>
<p><strong>结果</strong>：医院容量提升 128 倍，但每个病人的等待时间几乎不变！</p>
</div>

---

## 📋 目录

1. [为什么需要 MoE？](#why-moe)
2. [核心机制：Router + Expert](#mechanism)
3. [Tensor 维度变化全图解](#tensor-flow)
4. [Training Loop：梯度如何流动？](#training)
5. [Switch vs Mixtral：关键差异](#comparison)
6. [完整代码实现](#code)
7. [费曼自测题](#quiz)

---

<a id="why-moe"></a>
## 1. 为什么需要 MoE？

### 1.1 稠密模型的困境

传统 Transformer 的 FFN 层占据了 **2/3 的参数量**：

```python
class DenseFFN(nn.Module):
    def __init__(self, d_model=1024, d_ff=4096):
        self.W1 = nn.Linear(d_model, d_ff)   # [1024, 4096] = 4.2M 参数
        self.W2 = nn.Linear(d_ff, d_model)   # [4096, 1024] = 4.2M 参数
        # 总计 8.4M 参数，每个 token 都要过一遍
```

**问题**：
- GPT-3 (175B) 每次推理都激活 **100% 参数**
- 推理成本 ∝ 参数量 → 参数越多，推理越慢
- **扩展悖论**：想要更聪明，就必须更慢？

### 1.2 MoE 的核心洞察

<div class="insight-box">
<strong>💡 关键洞察</strong>：不同的 token 需要不同的"知识"来处理。
<ul>
<li>"The capital of France is" → 需要<strong>地理知识</strong></li>
<li>"def quicksort(arr):" → 需要<strong>编程知识</strong></li>
<li>"I feel so happy" → 需要<strong>情感理解</strong></li>
</ul>
<p>为什么要让同一个 FFN 处理所有这些？让专家各司其职！</p>
</div>

**MoE 的解决方案**：

| 对比项 | 稠密 FFN | MoE (128 Experts) |
|--------|----------|-------------------|
| 总参数量 | 8.4M | 8.4M × 128 = **1.075B** |
| 每次激活参数 | 8.4M (100%) | 8.4M (**0.78%**) |
| 模型容量 | 1× | **128×** |
| 推理成本 | 1× | **≈ 1×** |

**这就是 MoE 的魔法**：参数量暴涨 128 倍，推理成本几乎不变！

---

<a id="mechanism"></a>
## 2. 核心机制：Router + Expert

### 2.1 架构总览

```
输入 Token [B, L, D]
       ↓
   ┌─────────┐
   │ Router  │  ← 一个简单的 Linear 层
   └────┬────┘
        ↓ Softmax + Top-K
   ┌────┴────┐
   ↓    ↓    ↓
┌───┐ ┌───┐ ┌───┐
│E_0│ │E_1│ │...│ │E_127│  ← 128 个并行的 FFN
└─┬─┘ └─┬─┘ └───┘
  │     │
  └──┬──┘
     ↓ 加权求和
输出 [B, L, D]
```

### 2.2 Router：交通调度员

Router 就是一个 **Linear 层**，把每个 token 映射到 N 个 Expert 的概率分布：

```python
# Router 的全部代码
self.router = nn.Linear(d_model, num_experts, bias=False)  # [1024, 128]

# 前向传播
router_logits = self.router(x)           # [B, L, 128]
router_probs = F.softmax(router_logits, dim=-1)  # 概率分布
```

**具体例子**：

假设一个 token 的 embedding 是 `x = [0.1, 0.2, ..., 0.5]`（1024 维）

```python
# Router 计算
router_logits = W_router @ x  # [128]
# 假设结果是 [2.1, 0.5, 3.2, 0.1, ..., 0.8]

# Softmax 归一化
router_probs = softmax(router_logits)
# [0.15, 0.03, 0.45, 0.01, ..., 0.02]
#   ↑                ↑
# Expert 0: 15%   Expert 2: 45% ← 选这个！
```

### 2.3 Top-K 选择

**Switch Transformer (Top-1)**：每个 token 只选 **1 个** Expert

```python
topk_probs, topk_indices = torch.topk(router_probs, k=1)
# topk_indices = [2]  ← 选择 Expert 2
# topk_probs = [0.45] ← 权重 0.45
```

**Mixtral (Top-2)**：每个 token 选 **2 个** Expert

```python
topk_probs, topk_indices = torch.topk(router_probs, k=2)
# topk_indices = [2, 0]  ← 选择 Expert 2 和 Expert 0
# topk_probs = [0.45, 0.15] → 归一化 → [0.75, 0.25]
```

### 2.4 稀疏计算的实现

<div class="warning-box">
<strong>⚠️ 关键问题</strong>：128 个 Expert，每个 token 只用 1 个，怎么高效计算？
</div>

**朴素实现**（低效）：

```python
for expert_id in range(128):
    mask = (topk_indices == expert_id)
    if mask.any():
        output[mask] = experts[expert_id](x[mask])
```

**优化实现**（实际使用）：

```python
# 1. 按 Expert ID 重排所有 token
sorted_indices = topk_indices.argsort()
sorted_x = x[sorted_indices]

# 2. 批量处理每个 Expert 的 token
expert_outputs = []
for expert_id, expert in enumerate(experts):
    start, end = expert_boundaries[expert_id]
    if start < end:
        expert_outputs.append(expert(sorted_x[start:end]))

# 3. 还原到原始顺序
output = torch.cat(expert_outputs)[inverse_indices]
```

---

<a id="tensor-flow"></a>
## 3. Tensor 维度变化全图解

<div class="feynman-box">
<h4>🎨 可视化是建立直觉的最快方式</h4>
<p>让我们跟踪一个具体的例子，看 Tensor 如何流动。</p>
</div>

### 3.1 配置参数

```python
batch_size = 8
seq_len = 512
d_model = 1024
num_experts = 128
d_ff = 4096  # 每个 Expert 的 FFN 隐藏层
top_k = 1    # Switch Transformer
```

### 3.2 完整数据流

```
┌─────────────────────────────────────────────────────────────┐
│ Step 0: Input                                               │
│ x: [8, 512, 1024]                                          │
│ 含义: 8 个样本，每个 512 个 token，每个 token 1024 维        │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Router Logits                                       │
│ router_logits = W_r @ x                                     │
│ W_r: [1024, 128]                                           │
│ router_logits: [8, 512, 128]                               │
│ 含义: 每个 token 对 128 个 Expert 的"打分"                  │
└───────────────────────────┬─────────────────────────────────┘
                            ↓ Softmax
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Router Probabilities                                │
│ router_probs = softmax(router_logits, dim=-1)              │
│ router_probs: [8, 512, 128]                                │
│ 含义: 每个 token 选择各 Expert 的概率                        │
│ 每行和 = 1.0                                                │
└───────────────────────────┬─────────────────────────────────┘
                            ↓ Top-1
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Top-K Selection                                     │
│ topk_probs: [8, 512, 1]   ← 最大概率值                      │
│ topk_indices: [8, 512, 1] ← Expert ID (0-127)              │
│                                                             │
│ 例: Token[0,0] → Expert 42, 权重 0.67                       │
│     Token[0,1] → Expert 7,  权重 0.81                       │
│     Token[0,2] → Expert 42, 权重 0.55  ← 同一个 Expert!     │
└───────────────────────────┬─────────────────────────────────┘
                            ↓ Dispatch
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Expert Processing (稀疏!)                           │
│                                                             │
│ Expert 7:  收到 32 个 token  → [32, 1024]                   │
│ Expert 42: 收到 48 个 token  → [48, 1024]  ← 负载不均!      │
│ Expert 99: 收到 0 个 token   → 空闲                         │
│ ...                                                         │
│                                                             │
│ 每个 Expert 内部:                                            │
│ [N_i, 1024] → W1 → [N_i, 4096] → GELU → W2 → [N_i, 1024]   │
└───────────────────────────┬─────────────────────────────────┘
                            ↓ Weighted Combine
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Output                                              │
│ output[i] = topk_probs[i] × expert_output[i]               │
│ output: [8, 512, 1024]                                     │
│ 维度与输入完全一致!                                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 可视化：Router 决策热力图

![Router 决策与负载分布](/images/moe/01_router_decision_load.png)
*图 3.1: Router 决策热力图 (左)、Top-1 分配 (中)、负载分布 (右)*

**解读**：
- **左图**：每列是一个 token 对 16 个 Expert 的概率（颜色越深 = 概率越高）
- **中图**：每个 token 最终选择的 Expert（绿色 = 被选中）
- **右图**：Expert 负载分布
  - 🔴 红色柱子 = 过载（如 Expert 2 处理 12 个 token）
  - 🟠 橙色柱子 = 空闲（如 Expert 13 只处理 1 个）
  - 🟢 绿色虚线 = 理想均衡值

**这就是负载不均衡问题的根源！**

---

<a id="training"></a>
## 4. Training Loop：梯度如何流动？

### 4.1 完整的 Loss 函数

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{task}}}_{\text{Task Loss}} + \alpha \cdot \underbrace{\mathcal{L}_{\text{balance}}}_{\text{Load Balance}}
$$

其中 $\alpha = 0.01$（Switch Transformer 推荐值）

### 4.2 Load Balance Loss 推导

<div class="feynman-box">
<h4>🧠 费曼式理解 Load Balance Loss</h4>
<p>想象你是医院管理者，要让 128 个专家的工作量均衡：</p>
<ul>
<li><strong>f_i</strong> = Expert i 实际接诊的病人比例（频率）</li>
<li><strong>P_i</strong> = 护士给 Expert i 的平均推荐概率</li>
</ul>
<p>如果某个专家既<strong>实际接诊多</strong>（f 大），又<strong>被推荐概率高</strong>（P 大），说明系统在"偏袒"这个专家。惩罚它！</p>
</div>

**数学定义**：

$$
f_i = \frac{1}{B \cdot L} \sum_{b,l} \mathbb{1}[\text{Top1}(x_{b,l}) = i]
$$

$$
P_i = \frac{1}{B \cdot L} \sum_{b,l} \text{Router}(x_{b,l})_i
$$

$$
\mathcal{L}_{\text{balance}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

**具体计算例子**：

```python
# 假设 4 个 Expert，8 个 token
router_probs = torch.tensor([
    [0.7, 0.1, 0.1, 0.1],  # Token 0 → Expert 0 (p=0.7)
    [0.6, 0.2, 0.1, 0.1],  # Token 1 → Expert 0 (p=0.6)
    [0.5, 0.3, 0.1, 0.1],  # Token 2 → Expert 0 (p=0.5)
    [0.4, 0.4, 0.1, 0.1],  # Token 3 → Expert 0 (p=0.4)
    [0.1, 0.6, 0.2, 0.1],  # Token 4 → Expert 1
    [0.1, 0.1, 0.7, 0.1],  # Token 5 → Expert 2
    [0.1, 0.1, 0.1, 0.7],  # Token 6 → Expert 3
    [0.1, 0.1, 0.1, 0.7],  # Token 7 → Expert 3
])

# Top-1 选择结果: [0, 0, 0, 0, 1, 2, 3, 3]

# 计算 f (频率)
f = [4/8, 1/8, 1/8, 2/8]  # = [0.5, 0.125, 0.125, 0.25]

# 计算 P (平均概率)
P = router_probs.mean(dim=0)  # = [0.325, 0.2375, 0.1625, 0.2375]

# Load Balance Loss
N = 4
L_balance = N * sum(f[i] * P[i] for i in range(4))
# = 4 * (0.5×0.325 + 0.125×0.2375 + 0.125×0.1625 + 0.25×0.2375)
# = 4 * 0.2719 = 1.0876
```

**梯度分析**：

$$
\frac{\partial \mathcal{L}_{\text{balance}}}{\partial P_i} = N \cdot f_i
$$

- Expert 0: $\nabla P_0 = 4 \times 0.5 = 2.0$ ← 梯度最大，会被惩罚！
- Expert 1: $\nabla P_1 = 4 \times 0.125 = 0.5$ ← 梯度较小

→ 训练会**降低 Expert 0 的概率，提升其他 Expert 的概率**

### 4.3 梯度流动图解

![梯度流动路径](/images/moe/03_gradient_flow.png)
*图 4.1: MoE 反向传播的完整梯度流动路径*

**关键点**：

1. **Expert 参数的梯度**：
   - ✅ 被选中的 Expert 收到梯度
   - ❌ 未被选中的 Expert 梯度为 0

2. **Router 参数收到两种梯度**：
   - 🔵 来自主任务 Loss（通过 Gating Weight）
   - 🟠 来自 Load Balance Loss（直接作用于 softmax）

3. **梯度冲突**：
   - 主任务想让 Router **选择最好的 Expert**
   - LB Loss 想让 Router **均匀分配**
   - $\alpha = 0.01$ 平衡这两个目标

### 4.4 训练稳定性技巧

| 技巧 | 原因 | 代码 |
|------|------|------|
| **小初始化** | 防止 Router 一开始就偏向某些 Expert | `nn.init.normal_(router.weight, std=0.01)` |
| **Capacity Factor** | 限制每个 Expert 最多处理的 token 数 | `capacity = (B*L/N) * 1.25` |
| **BF16 训练** | 防止 softmax 上溢/下溢 | `model.to(torch.bfloat16)` |
| **梯度裁剪** | 防止梯度爆炸 | `clip_grad_norm_(params, 1.0)` |

---

<a id="comparison"></a>
## 5. Switch vs Mixtral：关键差异

| 特性 | Switch Transformer (2021) | Mixtral 8x7B (2023) |
|------|---------------------------|---------------------|
| **Top-K** | Top-1（极致稀疏） | Top-2（平衡性能） |
| **Expert 数量** | 2048（极端多） | 8（适合单机） |
| **每 token 计算量** | 1 个 Expert | 2 个 Expert |
| **负载均衡** | Auxiliary Loss | Token Choice + Expert Choice |
| **适用场景** | 预训练超大模型 | 指令微调 + 推理部署 |

![Switch vs Mixtral 架构对比](/images/moe/05_switch_vs_mixtral.png)
*图 5.1: Switch Transformer (Top-1) vs Mixtral (Top-2) 架构对比*

### 5.1 Mixtral 的 Top-2 优势

```python
# Top-2 路由
topk_probs, topk_indices = torch.topk(router_probs, k=2)
# 归一化
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

# 输出 = w1 * Expert1(x) + w2 * Expert2(x)
output = topk_probs[:, 0] * expert1_out + topk_probs[:, 1] * expert2_out
```

**优势**：
- 更好的**容错性**（如果 Top-1 过载，Top-2 补充）
- 负载**自然更均衡**
- 训练**更稳定**

**代价**：
- 推理成本 ×2（但仍远低于稠密模型：8 个 Expert 只激活 2 个 = 25%）

---

<a id="code"></a>
## 6. 完整代码实现

### 6.1 Switch MoE Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchMoELayer(nn.Module):
    """
    Switch Transformer 的 MoE Layer 实现
    
    参数:
        d_model: 输入/输出维度
        num_experts: Expert 数量
        d_ff: FFN 隐藏层维度
        capacity_factor: 容量因子（默认 1.25）
    """
    def __init__(self, d_model=1024, num_experts=128, d_ff=4096, 
                 capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        # Router: 简单的 Linear 层
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)  # 关键：小初始化
        
        # Experts: N 个独立的 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        B, L, D = x.shape
        
        # Step 1: Router 计算
        router_logits = self.router(x)  # [B, L, N]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Step 2: Top-1 选择
        topk_probs, topk_indices = torch.topk(router_probs, k=1, dim=-1)
        topk_probs = topk_probs.squeeze(-1)      # [B, L]
        topk_indices = topk_indices.squeeze(-1)  # [B, L]
        
        # Step 3: 容量限制
        capacity = int((B * L / self.num_experts) * self.capacity_factor)
        
        # Step 4: Expert 计算
        output = torch.zeros_like(x)
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        
        for expert_id in range(self.num_experts):
            mask = (topk_indices == expert_id)
            num_tokens = mask.sum().item()
            
            if num_tokens == 0:
                continue
            
            # 容量限制：只选概率最高的 token
            if num_tokens > capacity:
                masked_probs = torch.where(
                    mask, topk_probs, 
                    torch.tensor(-1e9, device=x.device)
                )
                _, top_indices = torch.topk(masked_probs.flatten(), k=capacity)
                new_mask = torch.zeros_like(mask.flatten(), dtype=torch.bool)
                new_mask[top_indices] = True
                mask = new_mask.view(B, L)
            
            # Expert 前向传播
            selected_x = x[mask]
            expert_out = self.experts[expert_id](selected_x)
            
            # 加权输出
            weights = topk_probs[mask].unsqueeze(-1)
            output[mask] = expert_out * weights
            expert_counts[expert_id] = mask.sum().item()
        
        # Step 5: 计算 Load Balance Loss
        f = expert_counts / (B * L)
        P = router_probs.mean(dim=[0, 1])
        load_balance_loss = self.num_experts * (f * P).sum()
        
        return output, load_balance_loss


# ===== 测试代码 =====
if __name__ == "__main__":
    moe = SwitchMoELayer(d_model=512, num_experts=8, d_ff=2048)
    x = torch.randn(2, 16, 512)
    
    output, lb_loss = moe(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"LB Loss: {lb_loss.item():.4f}")
    
    # 反向传播测试
    total_loss = output.sum() + 0.01 * lb_loss
    total_loss.backward()
    print(f"Router grad norm: {moe.router.weight.grad.norm():.4f}")
```

### 6.2 完整训练循环

```python
import torch.optim as optim

model = SwitchMoELayer(d_model=512, num_experts=16, d_ff=2048)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(10):
    x = torch.randn(8, 64, 512)
    target = torch.randn(8, 64, 512)
    
    output, lb_loss = model(x)
    main_loss = F.mse_loss(output, target)
    total_loss = main_loss + 0.01 * lb_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    print(f"Epoch {epoch}: Main={main_loss:.4f}, LB={lb_loss:.4f}")
```

**输出示例**：

```
Epoch 0: Main=1.0234, LB=1.4521
Epoch 1: Main=0.9876, LB=1.3214
...
Epoch 9: Main=0.7821, LB=1.0234  ← LB Loss 下降，负载更均衡！
```

---

<a id="quiz"></a>
## 7. 费曼自测题

<div class="quiz-box">
<h4>🧪 检验你是否真正理解了 MoE</h4>

**Q1**: 如果 MoE 有 128 个 Expert，每个 Expert 的 FFN 隐藏层是 4096，那么：
- 总参数量是多少？
- 每次推理激活多少参数？（Top-1）

<details>
<summary>点击查看答案</summary>

- 总参数：$2 \times 1024 \times 4096 \times 128 = 1.07B$
- 激活参数：$2 \times 1024 \times 4096 = 8.4M$（仅 0.78%）

</details>

---

**Q2**: Load Balance Loss 中，$f_i \cdot P_i$ 乘积的直觉含义是什么？

<details>
<summary>点击查看答案</summary>

- $f_i$ 是 Expert i 实际被选中的频率
- $P_i$ 是 Router 给 Expert i 的平均概率
- **乘积大**说明这个 Expert 既被选中多，又被推荐概率高 → 系统在"偏袒"它
- Loss 惩罚这种情况，迫使 Router 更均匀地分配

</details>

---

**Q3**: 为什么 Mixtral 选择 Top-2 而不是 Top-1？

<details>
<summary>点击查看答案</summary>

1. **容错性**：如果 Top-1 Expert 过载，Top-2 补充
2. **训练稳定性**：负载自然更均衡
3. **性能**：两个 Expert 的知识互补

代价：推理成本 ×2，但仍远低于稠密模型

</details>

---

**Q4**: 如果不使用 Load Balance Loss，会发生什么？

<details>
<summary>点击查看答案</summary>

- 部分 Expert 会被大量选择（马太效应）
- 部分 Expert 几乎不被使用 → 参数浪费
- 梯度更新不均 → 某些 Expert 过度训练，某些几乎不更新
- 极端情况：模型退化为只用 1-2 个 Expert 的"伪 MoE"

</details>
</div>

---

## 🔗 延伸阅读

1. **论文原文**：
   - [Switch Transformers (2021)](https://arxiv.org/abs/2101.03961)
   - [Mixtral of Experts (2023)](https://arxiv.org/abs/2401.04088)
   - [GShard (2020)](https://arxiv.org/abs/2006.16668) - 分布式 MoE

2. **开源实现**：
   - [Hugging Face Transformers](https://github.com/huggingface/transformers)
   - [Fairseq MoE](https://github.com/facebookresearch/fairseq)

3. **进阶话题**：
   - Expert Parallelism vs Tensor Parallelism
   - MoE + LoRA 稀疏微调
   - Dynamic Expert Selection

---

## 📊 总结

<div class="feynman-box">
<h4>🎯 一图总结 MoE</h4>

```
                    ┌─────────────────────────────────────┐
                    │           MoE 核心公式              │
                    │                                     │
                    │  y = Σ Router(x)_i × Expert_i(x)   │
                    │      i∈Top-K                       │
                    └─────────────────────────────────────┘
                                     ↓
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
   ┌────┴────┐                 ┌─────┴─────┐                ┌─────┴─────┐
   │ Router  │                 │  Experts  │                │   Loss    │
   │ (调度)  │                 │  (执行)   │                │  (均衡)   │
   └────┬────┘                 └─────┬─────┘                └─────┬─────┘
        │                            │                            │
   Softmax+TopK               并行 FFN 层                   f·P 惩罚项
   选择 K 个专家              稀疏激活                      防止偏袒
```

**记住三个数字**：
- **128×** 参数量提升
- **1%** 激活率（Top-1 with 128 Experts）
- **0.01** Load Balance Loss 权重

</div>

---

*Created: 2026-02-08 by Caius*
*Tags: #MoE #SwitchTransformer #Mixtral #DeepLearning #LLM*
