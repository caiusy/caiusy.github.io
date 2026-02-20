---
title: InstructGPT 与 RLHF 深度解析：费曼视角下的原理与实现
date: 2026-02-07 03:00:00
updated: 2026-02-07 03:00:00
mathjax: true
description: "一篇面向面试的深度长文。使用费曼技巧通俗解释 RLHF，结合详细的数学推导（SFT Loss, Bradley-Terry Model, PPO Clip, GAE），并深入分析每一阶段的输入输出流。"
categories:
  - 深度学习
tags:
  - Python
  - PyTorch
---

> 🎯 **面试官视角**：
> 这篇文章不仅仅是教程，更是你的**面试复习提纲**。
> 我们不堆砌代码，而是深入到 **原理 (Principles)**、**输入输出流 (I/O Flow)** 和 **数学推导 (Math Derivation)**。
> 如果你能用本文的逻辑把 RLHF 讲清楚，P7/L6 级别的算法岗面试不在话下。

<!-- more -->

---

# 0. 费曼时刻：什么是 Alignment？

想象你正在训练一只**天才鹦鹉**（GPT-3）。
它读遍了世界上的所有书，能补全任何句子。
- 你说：“床前明月光”，它接：“疑是地上霜”。
- 你问：“如何制造毒药？”，它接：“首先你需要……”（因为它在侦探小说里看过）。

**问题来了**：这只鹦鹉太“耿直”了，它不懂什么是**对**，什么是**错**，什么是**安全**。它只是在做**概率预测**。

**RLHF (Reinforcement Learning from Human Feedback)** 就是送这只鹦鹉去上**礼仪学校**的过程：
1.  **SFT（小学）**：老师给它标准答案，让它照着抄，学会“像人一样说话”。
2.  **RM（中考）**：老师不再给答案，而是给它的回答打分。它学会了“老师喜欢什么样的回答”。
3.  **PPO（体育特训）**：没有老师盯着，它自己根据之前的打分标准不断练习，试图拿高分，同时别把脑子练坏了（KL Penalty）。

---

# 1. 第一阶段：监督微调 (SFT)

> **一句话解释**：从“续写机器”变成“问答助手”。

### 1.1 原理深度解析

预训练模型（Pretrained Model）的目标是 {% raw %}$P(x_t|x_{<t})${% endraw %}，它倾向于**续写**。
SFT（Supervised Fine-Tuning）的目标是让模型学会**响应指令**。

虽然都是 Language Modeling，但核心区别在于 **Data Distribution（数据分布）** 和 **Loss Masking**。

### 1.2 输入输出流 (I/O Flow)

假设我们有一个 Batch 的数据。

*   **Input (Prompt)**: `x` = ["把这句话翻译成英文", "解释量子力学"]
*   **Target (Response)**: `y` = ["Translate this...", "Quantum mechanics is..."]

在 Tensor 层面：
*   **Input Tensor**: `[Batch, Seq_Len]` (例如 `[4, 1024]`)，包含了 `<Prompt> <Response> <EOS>`。
*   **Labels Tensor**: 形状同 Input。
    *   **关键点**：Prompt 部分的 Label 被设为 `-100` (PyTorch 中忽略计算 Loss 的标志)。
    *   只有 Response 部分参与梯度计算。

### 1.3 数学推导：MLE

{% raw %}
$$
\mathcal{L}_{\text{SFT}} = - \sum_{t \in \text{Response}} \log P_{\theta}(x_t \mid x_{<t})
$$
{% endraw %}

**面试考点**：
*   **Q: 为什么要 Mask 掉 Prompt 的 Loss？**
*   **A**: 因为 Prompt 是用户输入的，是已知的 Condition。我们不希望模型去“预测”用户会说什么，我们只希望模型在给定用户输入的情况下，预测正确的输出。如果不 Mask，模型会花费容量去记忆 Prompt 的分布，这是浪费。

---

# 2. 第二阶段：奖励模型 (RM)

> **一句话解释**：人类没办法给几亿条数据打分，所以我们训练一个“电子判官”来模仿人类的口味。

### 2.1 为什么是 Ranking 而不是 Scoring？

这是面试常考点。
*   **Scoring (打分)**：让标注员打 1-10 分。**缺点**：主观性太强。A 认为 7 分是好，B 认为 7 分是及格。数据噪声极大。
*   **Ranking (排序)**：给两个回答，选好的那个。**优点**：比较是容易的，一致性高。

### 2.2 输入输出流 (I/O Flow)

RM 通常是一个 BERT 或 GPT 架构的模型，但去掉了解码头，换成了一个**标量输出头 (Scalar Head)**。

*   **Input**: `(Prompt, Response_A, Response_B)`
*   **Process**:
    1.  把 `Prompt + Response_A` 喂给模型 $\rightarrow$ 得到标量 {% raw %}$r_A${% endraw %}。
    2.  把 `Prompt + Response_B` 喂给模型 $\rightarrow$ 得到标量 {% raw %}$r_B${% endraw %}。
*   **Output**: 两个分数 {% raw %}$r_A, r_B${% endraw %}。

### 2.3 数学推导：Bradley-Terry 模型

这是 RM 训练的核心数学基础。我们假设 $A$ 比 $B$ 好的概率服从 **Sigmoid** 分布：

{% raw %}
$$
P(A \succ B) = \sigma(r_A - r_B) = \frac{1}{1 + e^{-(r_A - r_B)}}
$$
{% endraw %}

**损失函数 (Log-Likelihood)**：
我们要最大化标注数据的似然度。假设数据告诉我们 A > B，则 Loss 为：

{% raw %}
$$
\begin{aligned}
\mathcal{L}_{\text{RM}} &= - \log P(A \succ B) \\
&= - \log \sigma(r_A - r_B) \\
&= - \log \left( \frac{1}{1 + e^{-(r_A - r_B)}} \right)
\end{aligned}
$$
{% endraw %}

**代码实现示例**：

```python
def ranking_loss(score_chosen, score_rejected):
    """
    score_chosen: shape [batch_size]
    score_rejected: shape [batch_size]
    """
    loss = -torch.nn.functional.logsigmoid(score_chosen - score_rejected).mean()
    return loss
```

---

# 3. 第三阶段：PPO 强化学习 (核心难点)

> **一句话解释**：左脚踩右脚上天。模型在 RM 的打分激励下，不断探索更高分的回答，同时被 KL 散度拉住，防止胡说八道。

这一块是面试的重灾区，涉及 **4 个模型** 和复杂的 **数学技巧**。

### 3.1 四个模型 (The Big Four)

在 PPO 训练时，显存中驻留着四个模型：

| 模型名称 | 角色 | 状态 | 作用 |
|:---|:---|:---|:---|
| **Actor** ({% raw %}$\pi_{\theta}${% endraw %}) | 运动员 | **Train** | 生成文本，我们要优化的对象 (初始化自 SFT) |
| **Critic** ({% raw %}$V_{\phi}${% endraw %}) | 教练 | **Train** | 估计状态价值 (Value Function)，用于计算 Advantage |
| **Ref Model** ({% raw %}$\pi_{\text{ref}}${% endraw %}) | 照妖镜 | Frozen | 提供基准概率，计算 KL 散度 (初始化自 SFT) |
| **Reward Model** ({% raw %}$R${% endraw %}) | 裁判 | Frozen | 给生成的文本打分 |

### 3.2 完整的 PPO 训练步 (Step-by-Step)

#### Step 1: Rollout (采样)
Actor 根据 Prompt 生成回答 `Response`。
同时，Ref Model 也计算生成该 `Response` 的概率。

#### Step 2: Calculate Reward (计算奖励)
真正的奖励不仅仅是 RM 的打分，还要减去 KL 惩罚。

{% raw %}
$$
R_{\text{total}} = R_{\text{RM}}(x, y) - \beta \cdot \log \left( \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} \right)
$$
{% endraw %}

**面试考点：为什么要 KL 惩罚？**
*   **为了防止 Reward Hacking**。RM 只是人类偏好的一个**拟合**（Proxy），它不是完美的。如果过度优化 RM 分数，模型会找到 RM 的漏洞（例如输出乱码但 RM 认为是好词）。KL 惩罚强制 Actor 不能偏离 SFT 模型太远，保证语言的流畅性和合理性。

#### Step 3: GAE (广义优势估计)
我们要计算 **Advantage (优势)**：当前的动作比“平均水平”好多少？

这里用到了 **Critic** 模型预测的 Value ({% raw %}$V(s)${% endraw %})。
TD Error ({% raw %}$\delta_t${% endraw %}) 定义为：
{% raw %}
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
{% endraw %}

GAE ({% raw %}$A_t${% endraw %}) 是 TD Error 的加权累加：
{% raw %}
$$
A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}
$$
{% endraw %}

#### Step 4: PPO Clip Update
这是 PPO 的精髓。我们希望更新策略，但**不能更新太猛**（Trust Region）。

定义概率比率 {% raw %}$ratio = \frac{\pi_{new}}{\pi_{old}}${% endraw %}。
目标函数：

{% raw %}
$$
L^{\text{CLIP}} = \min \left( ratio \cdot A_t, \text{clip}(ratio, 1-\epsilon, 1+\epsilon) \cdot A_t \right)
$$
{% endraw %}

*   如果 {% raw %}$A_t > 0${% endraw %}（好动作）：限制 {% raw %}$ratio${% endraw %} 最大为 {% raw %}$1+\epsilon${% endraw %}（奖励不要发过头）。
*   如果 {% raw %}$A_t < 0${% endraw %}（坏动作）：限制 {% raw %}$ratio${% endraw %} 最小为 {% raw %}$1-\epsilon${% endraw %}（惩罚不要太重）。

### 3.3 显存占用分析 (Memory Overhead)

假设一个 7B 模型占用 14GB 显存。PPO 训练需要多少显存？

1.  **Actor**: 14GB (参数) + 梯度 + 优化器状态 (Adam state 占 2倍参数)。
2.  **Critic**: 14GB (通常也是 7B，或者小一点) + 梯度 + 优化器状态。
3.  **Ref Model**: 14GB (Inference only)。
4.  **Reward Model**: 14GB (Inference only)。

加起来非常巨大！
**优化方案 (DeepSpeed/LoRA)**:
*   **Offload**: 把 Ref 和 RM 放到 CPU 内存里。
*   **LoRA**: Actor 和 Critic 只训练 Low-rank adapter，大幅减少优化器状态。
*   **Shared Backbone**: RM 和 Critic 共享一部分参数（Hydra Head）。

---

# 4. 代码实战：手写 PPO 核心逻辑

```python
import torch
import torch.nn.functional as F

def compute_ppo_loss(logprobs, old_logprobs, advantages, clip_eps=0.2):
    """
    logprobs: 当前策略生成的 token log 概率 [batch, seq_len]
    old_logprobs: 采样时策略生成的 token log 概率 (固定值)
    advantages: GAE 计算出的优势值
    """
    # 1. 计算概率比率 ratio = pi_new / pi_old
    # exp(log_a - log_b) = a / b
    ratio = torch.exp(logprobs - old_logprobs)
    
    # 2. 计算未截断的 Loss
    surr1 = ratio * advantages
    
    # 3. 计算截断后的 Loss
    # clamp 限制 ratio 在 [0.8, 1.2] 之间
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    
    # 4. 取最小值 (PPO Core: Pessimistic Bound)
    # 注意这里是 maximize objective，所以通常代码里是 return -loss.mean()
    loss = torch.min(surr1, surr2)
    
    return -loss.mean()
```

---

# 5. 总结与展望

### RLHF 的本质
RLHF 不是魔法，它是**用 Reward Model 作为一个可微的代理（Differentiable Proxy），把不可微的人类偏好（Human Preference）传递给语言模型**。

### 面试高频问题清单
1.  **SFT 阶段 Loss 怎么算的？** (Answer: Cross Entropy, Mask Prompt)
2.  **RM 为什么用 Ranking Loss？** (Answer: 归一化，抗噪声)
3.  **PPO 中 KL 散度的作用？** (Answer: 防止 Reward Hacking，保持分布一致性)
4.  **On-policy vs Off-policy？** (Answer: PPO 是 On-policy，每次更新都要重新采样，所以慢)

---
*本文最后更新于 2026-02-07 03:00:00*
