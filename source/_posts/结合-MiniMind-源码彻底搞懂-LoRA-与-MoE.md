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
description: 从 MiniMind 的真实实现出发，拆解 LoRA 的低秩适配、MoE 的路由与专家选择、辅助损失、张量形状、训练保存合并，以及两者组合时的工程细节。
review_status: published
---

大模型里经常同时出现两个缩写：LoRA 和 MoE。它们看起来都在优化模型，但优化的方向完全不同。

LoRA 关心的是：模型已经很大了，怎样用很少的可训练参数完成微调？

MoE 关心的是：怎样让模型拥有很多不同的专家能力，但每个 token 不必经过所有参数？

这篇文章不从概念图开始，而是直接沿着本项目 `vendor/minimind-master` 中的真实代码，把一次 token 的数据流走一遍。

<!-- more -->

![LoRA 原始 Linear 与低秩旁路结构图](/images/lora-moe-deep-dive/lora-diagram.svg)

## 一、先把两件事放回 Transformer

MiniMind 的一个 Block 可以抽象为：

```text
x
 ├── RMSNorm → Attention → Residual
 └── RMSNorm → FFN 或 MoE-FFN → Residual
```

Attention 负责 token 之间的信息交互，FFN 负责对每个 token 的特征做非线性变换。

在 `model_minimind.py` 中，关键选择只有一句：

```python
self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

所以 MoE 是模型结构层面的替换：开启后，Block 的 FFN 变成多个专家加一个 Router。

LoRA 则不替换 Block。它在模型创建完成后，通过 `apply_lora(model)` 给符合条件的 Linear 动态挂上旁路。

可以先记住这个判断：

```text
MoE 改变模型怎么计算
LoRA 改变模型怎么训练
```

## 二、LoRA 到底在近似什么

### 2.1 全量微调的代价

普通 Linear 层的计算是：

{% raw %}
$$
y = Wx
$$
{% endraw %}

全量微调时，权重变成：

{% raw %}
$$
W' = W + \Delta W
$$
{% endraw %}

其中 `W` 是预训练知识，`ΔW` 是当前任务要求的变化。全量微调意味着直接保存并更新完整的 `ΔW`。

LoRA 的经验假设是：对于一个具体任务，真正有用的变化并不需要覆盖整个高维空间，`ΔW` 可以用低秩矩阵近似：

{% raw %}
$$
\Delta W \approx BA
$$
{% endraw %}

于是：

{% raw %}
$$
y = Wx + BAx
$$
{% endraw %}

### 2.2 对应到项目代码

文件 `model/model_lora.py` 中的实现非常直接：

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.B(self.A(x))
```

如果原层是 `768 → 768`，完整权重有 `768 × 768 = 589824` 个参数。

若 `rank=16`，LoRA 只有：

```text
A: 768 × 16
B: 16 × 768
合计: 24576
```

这不是把 768 维输入直接压成一个标量，而是让任务变化先进入一个 16 维的子空间，再投影回 768 维。

### 2.3 为什么是两个矩阵，而不是一个小矩阵

`A` 学习输入方向，`B` 学习输出方向。它们的乘积 `BA` 的秩最多是 `r`：

{% raw %}
$$
\operatorname{rank}(BA) \le r
$$
{% endraw %}

所以 rank 是 LoRA 的容量旋钮：rank 越大，表达能力越强，参数和显存也越高；rank 太小，则可能无法表达任务变化。

### 2.4 初始化为什么不破坏基座模型

代码使用：

```python
self.A.weight.data.normal_(mean=0.0, std=0.02)
self.B.weight.data.zero_()
```

因为 `B=0`，刚挂载时：

{% raw %}
$$
BAx = 0
$$
{% endraw %}

因此新模型初始行为和原模型一致。训练开始后，LoRA 分支才逐渐学会修正。

## 三、项目里的 LoRA 究竟挂在哪些层

`apply_lora()` 的筛选条件是：

```python
if isinstance(module, nn.Linear) and module.in_features == module.out_features:
```

也就是只命中输入输出维度相同的 Linear。

以默认 hidden size 768、KV heads 4 为例：

| 层 | 形状 | 当前实现是否命中 |
| --- | --- | --- |
| `q_proj` | 768 → 768 | 是 |
| `k_proj` | 768 → 384 | 否 |
| `v_proj` | 768 → 384 | 否 |
| `o_proj` | 768 → 768 | 是 |
| `gate_proj` | 768 → intermediate | 通常否 |
| `up_proj` | 768 → intermediate | 通常否 |
| `down_proj` | intermediate → 768 | 通常否 |

因此这份实现的 LoRA 主要覆盖 Attention 的 `q_proj` 和 `o_proj`。这和很多生产实现不同，后者通常会显式配置目标模块，而不是只判断是否为方阵。

挂载后的 forward 相当于：

```python
def forward_with_lora(x, layer1=original_forward, layer2=lora):
    return layer1(x) + layer2(x)
```

注意这里用默认参数捕获 `original_forward` 和 `lora`，避免循环中的闭包引用被后续迭代覆盖。

## 四、LoRA 训练时到底更新什么

`train_lora.py` 中的冻结逻辑是：

```python
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
```

这带来三件事：

1. 基座权重不更新。
2. 梯度只保留在 LoRA 参数上。
3. AdamW 优化器只维护 LoRA 参数的状态。

保存时也只筛选 `.lora.` 参数。推理时，模型先加载基座，再调用 `apply_lora()`，最后加载 LoRA checkpoint。

如果追求推理速度，还可以合并：

```python
module.weight += module.lora.B.weight @ module.lora.A.weight
```

合并后的权重就是：

{% raw %}
$$
W_{merged}=W+BA
$$
{% endraw %}

合并后可以移除 LoRA 分支，推理计算图恢复成普通 Linear。

## 五、MoE：从一个 FFN 变成多个专家

![MoE Router 将 token 分发到不同 Expert](/images/lora-moe-deep-dive/moe-routing.svg)

### 5.1 普通 FFN

MiniMind 的普通 FFN 是 SwiGLU 形式：

```python
return self.down_proj(
    self.act_fn(self.gate_proj(x)) * self.up_proj(x)
)
```

它可以写成：

{% raw %}
$$
\operatorname{FFN}(x)=W_{down}\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right)
$$
{% endraw %}

所有 token 都经过同一个 FFN。

### 5.2 MoE FFN

`MOEFeedForward` 初始化时创建多个独立的 FFN：

```python
self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
self.experts = nn.ModuleList([
    FeedForward(config,
                intermediate_size=config.moe_intermediate_size)
    for _ in range(config.num_experts)
])
```

默认情况下：

```text
num_experts = 4
num_experts_per_tok = 1
```

也就是说，模型保存 4 个 FFN，但每个 token 默认只激活其中 1 个。

## 六、一次 MoE 前向传播的完整数据流

假设输入隐藏状态为：

```text
x: [batch, seq_len, hidden_size]
```

代码先把它展平：

```python
x_flat = x.view(-1, hidden_dim)
```

形状变成：

```text
x_flat: [batch × seq_len, hidden_size]
```

这样 Router 可以把每一个 token 独立路由。

### 6.1 Router 打分

```python
scores = F.softmax(self.gate(x_flat), dim=-1)
```

如果有 4 个专家：

```text
self.gate(x_flat): [batch × seq_len, 4]
scores:             [batch × seq_len, 4]
```

某个 token 的分数可能是：

```text
[0.05, 0.10, 0.80, 0.05]
```

### 6.2 Top-K 选择

```python
topk_weight, topk_idx = torch.topk(
    scores,
    k=self.config.num_experts_per_tok,
    dim=-1,
    sorted=False
)
```

Top-1 时，上面的 token 会得到：

```text
topk_idx    = [2]
topk_weight = [0.80]
```

Top-2 时，可能得到专家 2 和专家 1，并把两个专家的输出按权重相加：

{% raw %}
$$
y=w_1E_1(x)+w_2E_2(x)
$$
{% endraw %}

代码还会归一化 Top-K 权重，确保选中的专家权重和为 1：

```python
topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
```

### 6.3 分发 token 并聚合结果

核心循环是：

```python
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(
            0,
            token_idx,
            (expert(x_flat[token_idx]) * weight).to(y.dtype)
        )
```

假设有 8 个 token，路由结果是：

```text
Expert 0: token 1, 6
Expert 1: token 0, 4, 7
Expert 2: token 2, 3
Expert 3: token 5
```

那么 Expert 0 不会处理 token 0、2、3 等不属于它的输入。各专家只计算自己收到的 token，最后通过 `index_add_` 写回原 token 位置。

## 七、MoE 的参数量与计算量

假设普通模型有一个 FFN，MoE 有 4 个同规模 Expert：

```text
参数容量：约增加到 4 倍
Top-1 每 token 的专家计算：仍约为 1 个 FFN
```

但这不代表总成本完全不变。

Router 本身要计算，token 需要分发，专家负载可能不均匀，所有专家参数也必须存储。在多卡系统中，还可能发生跨卡通信。

所以 MoE 的准确表述应该是：

```text
用稀疏激活换取更大的参数容量
```

而不是参数量增加后计算量完全不变。

## 八、为什么必须有 auxiliary loss

Router 容易出现塌缩：几乎所有 token 都选择同一个专家。代码中使用实际负载和平均路由概率构造辅助损失：

```python
load = F.one_hot(
    topk_idx,
    self.config.num_experts
).float().mean(0)

self.aux_loss = (
    load * scores.mean(0)
).sum() * self.config.num_experts \\
                * self.config.router_aux_loss_coef
```

设第 `i` 个专家的实际 token 比例为 `f_i`，Router 平均概率为 `P_i`，则核心形式可以理解为：

{% raw %}
$$
L_{aux}=N\sum_{i=1}^{N}f_iP_i
$$
{% endraw %}

如果某个专家既被大量选中，又长期得到很高概率，它对辅助损失的贡献就会变大，训练会推动 Router 改善负载分布。

本项目还把 `router_aux_loss_coef` 默认设为 `5e-4`，说明它是约束项，不应压过语言建模的主损失。

在训练循环中，MoE 的总目标通常是：

{% raw %}
$$
L=L_{CE}+L_{aux}
$$
{% endraw %}

非 MoE 模式下，模型将 `aux_loss` 设置为零标量，因此训练流程可以统一处理。

## 九、MoE 中一个容易忽略的工程细节

代码里有这一段：

```python
elif self.training:
    y[0, 0] += 0 * sum(
        p.sum() for p in expert.parameters()
    )
```

它数值上等于加零，但把没有被当前 batch 选中的 Expert 参数接入了计算图。

原因是分布式训练时，如果某些专家在一个 batch 中完全没有 token，框架可能把它们判断为 unused parameters，进而影响 DDP 的梯度同步。这个写法不改变前向数值，却保留了图连接。

## 十、LoRA 与 MoE 能不能一起用

可以，但要区分两层含义：

```text
MoE 决定有哪些专家，以及 token 走哪个专家
LoRA 决定哪些权重用低秩方式适配新任务
```

组合后可以是：

```text
Attention
 ├── q_proj + LoRA
 └── o_proj + LoRA

MoE FFN
 ├── Router
 ├── Expert 0
 ├── Expert 1
 ├── Expert 2
 └── Expert 3
```

但当前项目的 `apply_lora()` 只选择方阵 Linear。MoE 的 Router 是：

```text
hidden_size → num_experts
```

Expert 的投影通常是：

```text
hidden_size → intermediate_size
intermediate_size → hidden_size
```

这些层大多不是方阵，因此当前实现不会自动给 Expert 投影加 LoRA。

如果要做专家 LoRA，应改成显式目标模块匹配，例如匹配 `experts.*.gate_proj`、`experts.*.up_proj` 和 `experts.*.down_proj`，并单独决定是否训练 Router。

Router 是否训练是一个关键选择：

```text
冻结 Router：保持原有专家分工，训练更稳定
训练 Router：允许任务重新定义 token 到专家的分配，但更容易负载失衡
```

## 十一、在本项目中如何选择

![LoRA 与 MoE 的目标和实现方式对比](/images/lora-moe-deep-dive/comparison.svg)

如果你的目标是让 MiniMind 学会新的回答格式、工具调用格式或领域表达，优先考虑 LoRA。它改动小，checkpoint 小，训练成本低。

如果你的目标是提高模型容量，允许不同 token 使用不同的 FFN 能力，才考虑 MoE。它需要从预训练或完整训练阶段就处理好专家负载、checkpoint 结构和推理部署。

一个实用路线是：

```text
先用 dense 模型验证数据和任务
 ↓
用 LoRA 快速验证微调收益
 ↓
任务规模和能力边界确认后，再评估 MoE
```

特别提醒：`use_moe=1` 不是一个可以随时打开的推理开关。MoE 模型的权重结构不同，必须加载对应的 MoE checkpoint。普通 dense 权重和 MoE 权重不能直接混用。

## 十二、最后把源码压缩成四个公式

LoRA 的旁路：

{% raw %}
$$
y=Wx+BAx
$$
{% endraw %}

低秩约束：

{% raw %}
$$
\operatorname{rank}(BA)\le r
$$
{% endraw %}

MoE 的稀疏混合：

{% raw %}
$$
y=\sum_{i\in TopK(x)}w_iE_i(x)
$$
{% endraw %}

MoE 的训练目标：

{% raw %}
$$
L=L_{CE}+L_{aux}
$$
{% endraw %}

所以，LoRA 是在原模型旁边学习一个低秩补丁，MoE 是在原 FFN 位置建立一组专家并让 Router 动态分流。

一个负责少训练参数，一个负责多模型容量。理解这两个差别，后面再看 QLoRA、Mixtral、专家并行或 MoE-LoRA，就不会只停留在名词层面了。
