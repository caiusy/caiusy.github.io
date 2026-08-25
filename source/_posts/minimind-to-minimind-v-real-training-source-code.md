---
title: 从 MiniMind 到 MiniMind-V：Pretrain、SFT、视觉 Token 与 Loss 的源码级推导
date: 2026-08-25 22:10:00
updated: 2026-08-25 23:48:53
mathjax: true
description: >-
  从 MiniMind 与 MiniMind-V 的真实源码出发，对比语言模型 Pretrain/SFT、VLM Pretrain/SFT、视觉 token 注入、冻结策略、梯度边界与 next-token cross-entropy，并用最终 checkpoint 对一张真实图片完成逐维前向和数值复算。
categories:
  - AI与大模型
  - 深度学习
tags: [MiniMind, MiniMind-V, Vision-Language-Model, Pretrain, SFT, DPO, PPO, GRPO, SigLIP2]
type: deep-dive
difficulty: advanced
review_status: published
cover: /images/minimind-v-real-training/00-cover-signal-cartography.png
---

MiniMind-V 的关键并不是“给 MiniMind 加一张图片”，而是完成一次严格的坐标变换：像素先变成 64 个视觉 patch token，再投影到 MiniMind 的 768 维 hidden space，最后占据文本序列中的 64 个位置。进入第一个 Transformer block 后，模型看到的已经不再是“图片”和“文字”，而是一条统一的 768 维向量序列。

本文不以环境安装和训练日志为主线，而是回答四个技术问题：

1. MiniMind 文本 Pretrain 与 SFT 到底改变了什么？
2. MiniMind 与 MiniMind-V 的网络、token 和参数路径有什么结构性差异？
3. MiniMind-V Pretrain 与 SFT 明明调用同一个 forward，为什么仍是两个不同阶段？
4. 一张真实图片如何产生真实 logits，并最终得到可手工复算的 loss？

所有源码默认值来自固定 commit；所有 shape、token 数值和 loss 来自远端最终 SFT checkpoint 的只读 forward。远端本次没有执行 VLM Pretrain，但这不影响我们依据真实源码完整分析该阶段。

![从真实图片到视觉 Token 与语言 Token 的技术图谱](/images/minimind-v-real-training/00-cover-signal-cartography.png)

<!-- more -->

## 1. 两个坐标轴：sequence length 与 feature dimension

理解 VLM 最常见的障碍，是把“token 数”和“token 的特征维度”混成一件事。

| 符号 | 含义 | 本文模型中的值 |
|---|---|---:|
| $B$ | batch size | 训练实参决定 |
| $L$ | 序列位置数 | 文本 PT 340；文本/VLM SFT 768；VLM PT 450 |
| $H$ | 每个 token 的 hidden dimension | 768 |
| $V$ | 文本词表大小 | 6,400 |
| $N_v$ | 单图视觉 token 数 | 64 |
| $I$ | SwiGLU intermediate dimension | 2,432 |
| $n_q/n_{kv}$ | query / KV heads | 8 / 4 |
| $d_h$ | head dimension | 96 |

语言模型处理的主张量是：

{% raw %}
$$
X\in\mathbb R^{B\times L\times H}.
$$
{% endraw %}

$L$ 是横向 token 序列，$H$ 是每个 token 内部的特征。MiniMind-V 的 64 个视觉 token 占用 $L$ 轴上的 64 个位置；每个视觉 token 仍然有 $H=768$ 个连续特征。它不是给模型额外增加一个“图片维度”。

最终 LM Head 只把最后一维从 $H$ 映射到词表 $V$：

{% raw %}
$$
[B,L,768]\;W_{\mathrm{lm}}\longrightarrow[B,L,6400].
$$
{% endraw %}

所以 MiniMind-V 的输出目标仍然是文本 next-token prediction。图片只改变条件上下文，不改变输出词表。

## 2. 两张总表：模型差异与训练差异

### 2.1 MiniMind 与 MiniMind-V 的结构对照

![MiniMind 与 MiniMind-V 共享语言主干但具有不同输入接口](/images/minimind-v-real-training/01-minimind-vs-minimind.png)

| 比较项 | MiniMind | MiniMind-V |
|---|---|---|
| 原始输入 | 文本字符串 | 图片二进制 + 多轮对话 |
| Tokenizer 输出 | `input_ids [B,L]` | `input_ids [B,L]`，含连续 64 个 image-pad id |
| 额外输入 | 无 | `pixel_values [B,3,256,256]` |
| 输入 embedding | `[B,L,768]` | 先查文本 embedding，再替换 64 个槽位 |
| 视觉编码器 | 无 | frozen SigLIP2，输出 `[B,64,768]` |
| 跨模态接口 | 无 | `LN → Linear → GELU → Linear` Projector |
| 融合方式 | 不涉及 | embedding replacement；没有 cross-attention |
| 语言主干 | 8×MiniMind block | 完全相同的 8×MiniMind block |
| 输出 logits | `[B,L,6400]` | `[B,L,6400]` |
| 基础 loss | next-token CE | next-token CE |
| 运行时唯一参数 | 63.912192M | 159.647232M，其中 94.552320M vision 始终冻结 |

MiniMind-V 直接继承 `MiniMindForCausalLM`：

```python
class MiniMindVLM(MiniMindForCausalLM):
    def __init__(self, config, vision_model_path=...):
        super().__init__(config)
        self.vision_encoder, self.processor = self.get_vision_model(...)
        self.vision_proj = MMVisionProjector(
            config.image_hidden_size,
            config.hidden_size,
        )
```

结构差异全部发生在第一个语言 block 之前。后续 GQA、RoPE、RMSNorm、SwiGLU、残差与 LM Head 都与 MiniMind 共用同一实现。

### 2.2 四种 Pretrain/SFT 的核心对照

| 项目 | MiniMind PT | MiniMind SFT | MiniMind-V PT | MiniMind-V SFT |
|---|---|---|---|---|
| 脚本 | `train_pretrain.py` | `train_full_sft.py` | `train_pretrain_vlm.py` | `train_sft_vlm.py` |
| 默认起点 | `none` | `pretrain` | `llm` | `pretrain_vlm` |
| 数据 | 普通文本 | 多轮指令对话 | image-caption 对话 | 图像问答 + caption + 纯文本 |
| 默认 $L$ | 340 | 768 | 450 | 768 |
| 默认 batch | 32 | 16 | 16 | 4 |
| 每个 microbatch 的序列槽位 $B\times L$ | 10,880 | 12,288 | 7,200 | 3,072 |
| 单图占用的视觉槽位 | 0 | 0 | 64 | 64 |
| 默认 LR | `5e-4` | `1e-5` | `4e-4` | `5e-6` |
| 监督位置 | 全部非 pad token | assistant token | assistant caption token | assistant response token |
| 默认可训练参数 | 全部 MiniMind | 全部 MiniMind | 仅 Projector，1.182720M | Projector + Block 0 + Block 7，15.931776M |
| loss | CE | CE | CE | CE |
| 学习目标 | 建模文本分布 | 学会遵循指令 | 视觉→语言对齐 | 图像条件下遵循指令 |

同一个 CE 在不同的数据分布、label mask 和可训练参数集合上，代表四个不同的优化问题。

### 2.3 Token length：容量、padding 宽度与有效监督长度不是同一个量

本文会同时出现四种“长度”：

| 长度 | 定义 | 本文例子 |
|---|---|---|
| 模型位置容量 | RoPE 可以索引的最大位置 | `max_position_embeddings=32768` |
| 训练序列宽度 | Dataset 截断并 padding 到的固定 $L$ | VLM SFT 为 768 |
| 非 padding 长度 | 一条样本实际写入的 token 数 | 固定真实样本为 288 |
| 有效监督长度 | `labels != -100` 的 target 数 | 固定真实样本为 189 |

因此真实样本的 768 个位置可以拆成：

```text
固定 tensor width               768
├─ 非 padding token             288
│  ├─ 图像槽位                   64  (positions 4..67)
│  ├─ 模板、用户问题等上下文       35
│  └─ assistant 监督 token       189 (positions 99..287)
└─ padding / ignored            480
```

这里的 64 个视觉 token 已经包含在 288 个非 padding token 和 $L=768$ 中，不是在序列之外额外增加 64。训练时计算图仍按 padding 后的 $B\times L$ 构建，所以短样本也会消耗完整 768 宽度的 attention、hidden state 和 logits 内存。

文本 Pretrain 默认还有 `accumulation_steps=8`：每个 microbatch 有 $32\times340=10,880$ 个序列位置，每次 optimizer update 累积约 $87,040$ 个位置。梯度累积增加有效 batch token 数，但不会让八个 microbatch 的 activation 同时驻留显存。

### 2.4 内存：参数、activation、logits 与 KV Cache 分开计算

下面是理论张量内存，不是 `nvidia-smi` 的进程占用。实际显存还包括 CUDA context、PyTorch allocator、临时 kernel workspace、未列出的 autograd 保存张量和碎片。

任意 tensor 的基础公式是：

{% raw %}
$$
M_{\mathrm{tensor}}
=\left(\prod_i d_i\right)\times s_{\mathrm{dtype}},
$$
{% endraw %}

其中 BF16/FP16 的 $s=2$ bytes，FP32 的 $s=4$ bytes。

本次 VLM SFT 使用 $B=4,L=768,H=768,V=6400$。几个关键 tensor 的单体大小为：

| Tensor | Shape / 公式 | dtype 假设 | 大小 |
|---|---|---:|---:|
| 输入图片 | `[4,3,256,256]` | FP32 | 3.000 MiB |
| SigLIP2 输出 | `[4,64,768]` | BF16 | 0.375 MiB |
| Projector 输出 | `[4,64,768]` | BF16 | 0.375 MiB |
| 一份 hidden state | `[4,768,768]` | BF16 | 4.500 MiB |
| Q projection | `[4,768,768]` | BF16 | 4.500 MiB |
| K 或 V projection | `[4,768,384]` | BF16 | 2.250 MiB |
| 一份 gate/up activation | `[4,768,2432]` | BF16 | 14.250 MiB |
| 完整 logits | `[4,768,6400]` | BF16 | 37.500 MiB |

这解释了为什么 sequence length 和词表大小对显存特别敏感：hidden/FFN activation 近似随 $B\times L$ 线性增长，而显式 attention score 在非 fused 实现中会随 $B\times L^2$ 增长；logits 则随 $B\times L\times V$ 增长。当前实现使用 scaled-dot-product attention，可避免长期保存完整手写 attention matrix，但反向传播仍需保存多组中间量。

参数与优化器状态需要单独统计。本次运行时唯一参数为 159.647232M；若以 FP32 常驻：

| 项目 | 计算 | 理论大小 |
|---|---:|---:|
| 全部模型参数 | `159,647,232 × 4` | 609.006 MiB |
| 可训练参数梯度 | `15,931,776 × 4` | 60.775 MiB |
| AdamW 一阶、二阶矩 | `15,931,776 × 2 × 4` | 121.550 MiB |
| 参数 + trainable grad + Adam moments | 上述三项 | 791.331 MiB |

这里没有为冻结参数计算梯度和 Adam moments。PyTorch AdamW 的 state 是懒初始化的；虽然 SFT 脚本把全部参数传入 optimizer，`requires_grad=False` 的参数没有梯度，也不会获得对应的 moment tensor。视觉编码器还在 `torch.no_grad()` 中执行，因此不会保存视觉主干的反向图，这是冻结 SigLIP2 能显著节约训练内存的原因。

推理阶段主要关注 KV Cache。对 GQA 模型，每层的 K/V cache 为：

{% raw %}
$$
M_{\mathrm{KV/layer}}
=2\times B\times L\times n_{kv}\times d_h\times s.
$$
{% endraw %}

8 层合计：

{% raw %}
$$
M_{\mathrm{KV,total}}
=2\times B\times L\times4\times96\times2\times8.
$$
{% endraw %}

在 $B=1$、BF16 下，$L=768$ 的 KV Cache 正好是 9 MiB；若真正使用到位置容量 $L=32768$，则为 384 MiB。GQA 只缓存 4 个 KV heads；如果像标准 MHA 那样缓存 8 个 heads，KV 内存会翻倍。

## 3. MiniMind 训练路线：共享网络，逐步改变监督信号

![MiniMind Pretrain、SFT 与三类对齐训练的分叉关系](/images/minimind-v-real-training/02-training-stage-map.png)

当前仓库更准确的默认关系是：

```text
Pretrain → SFT ┬→ DPO
               ├→ PPO
               └→ GRPO / CISPO
```

`train_dpo.py`、`train_ppo.py` 和 `train_grpo.py` 默认都从 `full_sft` 初始化，因此三者是 SFT 后的替代路线，不是必须连续执行的关卡。

### 3.1 文本 Pretrain：每个有效 token 都是训练样本

`PretrainDataset`：

```python
tokens = tokenizer(text, max_length=max_length - 2).input_ids
tokens = [bos_id] + tokens + [eos_id]
input_ids = tokens + [pad_id] * (max_length - len(tokens))
labels = input_ids.clone()
labels[input_ids == pad_id] = -100
```

若有效 target 集合为 $\mathcal V$：

{% raw %}
$$
\mathcal L_{\mathrm{LM-PT}}
=-\frac{1}{|\mathcal V|}
\sum_{t\in\mathcal V}
\log p_\theta(x_t\mid x_{\lt t}).
$$
{% endraw %}

正文、标点、事实、语法和篇章衔接都产生梯度。模型学习“自然文本的下一个 token 通常是什么”。

### 3.2 文本 SFT：上下文全部可见，只处罚 assistant

SFT 通过 chat template 串起 system、user、assistant，再寻找 assistant span：

```python
labels = [-100] * len(input_ids)
if input_ids[i:i + len(bos_id)] == bos_id:
    start = i + len(bos_id)
    ...
    for j in range(start, end + len(eos_id)):
        labels[j] = input_ids[j]
```

用户问题没有从 forward 中删除，仍参与 causal attention；只是它的 target label 为 `-100`，不直接进入 CE。

{% raw %}
$$
\mathcal L_{\mathrm{LM-SFT}}
=-\frac{1}{|\mathcal A|}
\sum_{t\in\mathcal A}
\log p_\theta(y_t\mid y_{\lt t},x_{\mathrm{system}},x_{\mathrm{user}}).
$$
{% endraw %}

Pretrain 与 SFT 都经过同一个网络：

```text
input_ids       [B,L]
token embedding [B,L,768]
8× Transformer  [B,L,768]
LM Head          [B,L,6400]
labels           [B,L]
```

默认 $L$ 从 340 增至 768，只改变 batch 的截断/padding 宽度；`H=768`、`V=6400`、heads 和 FFN 维度都不变。`max_position_embeddings=32768` 是模型位置容量，也不等于实际训练 $L$。

### 3.3 DPO、PPO、GRPO 改变的是优化对象

DPO 比较 chosen 与 rejected 相对 reference 的优势：

{% raw %}
$$
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$
{% endraw %}

PPO 使用 Reward Model、Critic 与 GAE：

{% raw %}
$$
\delta_t=r_t+\gamma V_{t+1}-V_t,
\qquad
A_t=\delta_t+\gamma\lambda A_{t+1}.
$$
{% endraw %}

GRPO 用同一 prompt 的一组回答做 reward 标准化，从而不需要 Critic：

{% raw %}
$$
A_i=\frac{R_i-\mu_{\mathrm{group}}}
{\sigma_{\mathrm{group}}+10^{-4}}.
$$
{% endraw %}

当前 `train_grpo.py` 默认 `loss_type=cispo`；只有显式传入 `--loss_type grpo` 才走标准 GRPO clip 分支。

## 4. MiniMind 主干：视觉模型最终也进入这 8 个 Block

MiniMind dense 配置：

```python
vocab_size              = 6400
hidden_size             = 768
num_hidden_layers       = 8
num_attention_heads     = 8
num_key_value_heads     = 4
head_dim                = 96
intermediate_size       = 2432
max_position_embeddings = 32768
```

真实线性层 shape：

```text
Q: [B,L,768] → [B,L,768] → [B,L,8,96]
K: [B,L,768] → [B,L,384] → [B,L,4,96]
V: [B,L,768] → [B,L,384] → [B,L,4,96]
gate/up: [B,L,768]  → [B,L,2432]
down:    [B,L,2432] → [B,L,768]
```

{% raw %}
$$
Y=X+\operatorname{Attention}(\operatorname{RMSNorm}(X)),
$$
{% endraw %}

{% raw %}
$$
Z=Y+W_{\mathrm{down}}\left[
\operatorname{SiLU}(W_{\mathrm{gate}}\operatorname{RMSNorm}(Y))
\odot W_{\mathrm{up}}\operatorname{RMSNorm}(Y)
\right].
$$
{% endraw %}

每个 block 都保持 `[B,L,768]`。视觉 token 只要被 Projector 变成 768 维，就能直接复用语言主干。

## 5. MiniMind-V：像素如何写进文本序列

### 5.1 视觉编码器与 Projector

`VLMConfig` 固定：

```python
image_hidden_size = 768
image_token_len   = 64
image_ids         = [12]
```

Projector：

```python
self.mlp = nn.Sequential(
    nn.LayerNorm(768),
    nn.Linear(768, 768),
    nn.GELU(),
    nn.Linear(768, 768),
)
```

{% raw %}
$$
[B,64,768]\longrightarrow[B,64,768].
$$
{% endraw %}

它不改变 token 数和维度，而是改变坐标系：把 SigLIP2 表示变成语言模型可使用的 hidden state。

### 5.2 P32 为什么正好得到 64 个 token

真实图片经 processor 变成 `[1,3,256,256]`，patch size 为 32：

{% raw %}
$$
N_v=\frac{256}{32}\times\frac{256}{32}=8\times8=64.
$$
{% endraw %}

![真实图片、P32 Patch、64 个视觉 Token 与融合序列](/images/minimind-v-real-training/04-real-image-token-journey.png)

### 5.3 image-pad 是槽位，不是视觉词表

```python
self.image_special_token = image_special_token * image_token_len
content = content.replace('<image>', self.image_special_token)
```

真实样本中 image-pad token id 为 12，位于 positions `4..67`。Tokenizer 先查普通文本 embedding，随后 `count_vision_proj` 替换这一连续区间：

```python
hb = torch.cat(
    (hb[:start], vf[b][k][:i - start], hb[i:]),
    dim=0,
)[:seqlen]
```

这不是离散视觉 tokenizer，不会给 6,400 词表增加图片类别，也没有 cross-attention；它直接替换 `inputs_embeds` 的 64 个位置。

### 5.4 完整 forward

```python
hidden_states = embed_tokens(input_ids)          # [B,L,768]
vision = vision_encoder(pixel_values)             # [B,64,768]
vision = vision_proj(vision)                      # [B,64,768]
hidden_states = count_vision_proj(..., vision)    # [B,L,768]

for layer in model.layers:
    hidden_states, present = layer(hidden_states, ...)

logits = lm_head(norm(hidden_states))             # [B,L,6400]
```

后面的文本 token 可以通过 causal attention 关注前面的视觉槽位；视觉槽位看不到其后的答案 token，仍保持标准自回归方向。

## 6. MiniMind-V Pretrain 与 SFT：forward 相同，梯度边界不同

![MiniMind-V Pretrain 与 SFT 的源码级差异](/images/minimind-v-real-training/03-vlm-pretrain-vs-sft.png)

### 6.1 两个脚本共享训练循环

```python
model, tokenizer, preprocess = init_vlm_model(...)
train_ds = VLMDataset(...)
res = model(input_ids, labels=labels, pixel_values=pixel_values)
loss = res.loss + res.aux_loss
loss.backward()
```

两者没有分别定义“视觉对比 loss”和“指令 loss”。dense 模型 `aux_loss=0`；Projector 参数上的乘零项只保证 DDP 梯度图完整。

### 6.2 源码 diff 真正改变七件事

| 参数 | VLM Pretrain | VLM SFT | 含义 |
|---|---:|---:|---|
| `save_weight` | `pretrain_vlm` | `sft_vlm` | checkpoint 身份 |
| `batch_size` | 16 | 4 | SFT 更长且训练层更多 |
| `learning_rate` | `4e-4` | `5e-6` | SFT LR 是 PT 的 1/80 |
| `max_seq_len` | 450 | 768 | SFT 容纳问题与长回答 |
| `data_path` | `pretrain_i2t.parquet` | `sft_i2t.parquet` | 数据分布不同 |
| `from_weight` | `llm` | `pretrain_vlm` | 初始化链不同 |
| `freeze_llm` | 2 | 1 | 梯度进入哪些参数 |

还有一个实现差异：Pretrain 的 AdamW 显式过滤 `requires_grad=True` 参数，SFT 则传入 `model.parameters()`。冻结参数没有 grad，因此后者也不会实际更新它们。

### 6.3 数据类相同，数据分布不同

两阶段都使用 `VLMDataset.generate_labels()`，因此都只监督 assistant span。VLM Pretrain 的“Pretrain”并不等于文本 Pretrain 的 all-token label；它仍是 assistant-only caption generation。

{% raw %}
$$
\mathcal L
=-\frac{1}{|\mathcal A|}
\sum_{t\in\mathcal A}
\log p_\theta(y_t\mid y_{\lt t},x_{\mathrm{image}},x_{\mathrm{user}}).
$$
{% endraw %}

Pretrain 数据主要要求“识别并描述”；SFT 数据要求“理解问题、选择视觉证据并按指令组织回答”。相同 loss 不代表相同任务。

### 6.4 `freeze_llm=2`：Pretrain 只训练 Projector

```python
for name, param in model.named_parameters():
    if 'vision_proj' not in name:
        param.requires_grad = False
```

令视觉编码器、Projector、语言模型参数为 $\theta_v,\theta_p,\theta_l$：

{% raw %}
$$
\nabla_{\theta_v}\mathcal L=0,
\qquad
\nabla_{\theta_l}\mathcal L=0,
\qquad
\nabla_{\theta_p}\mathcal L\ne0.
$$
{% endraw %}

只有 `1.182720M` Projector 参数更新。语言模型是冻结的目标坐标系；Projector 必须把视觉向量送到能让现有语言模型预测 caption 的区域。因为随机 Projector 需要较大移动且不会破坏语言权重，默认 LR 可设为 `4e-4`。

### 6.5 `freeze_llm=1`：SFT 打开首尾语言 block

```python
last_idx = num_hidden_layers - 1
for name, param in model.model.named_parameters():
    if 'layers.0.' in name or f'layers.{last_idx}.' in name:
        param.requires_grad = True
```

| 模块 | 参数量 | 作用 |
|---|---:|---|
| Projector | 1.182720M | 调整视觉→语言接口 |
| Block 0 | 7.374528M | 适应混合视觉/文本 embedding |
| Block 7 | 7.374528M | 适应视觉条件下回答分布 |
| 合计 | 15.931776M | 约为运行时总参数的 9.98% |

{% raw %}
$$
\nabla_{\theta_v}\mathcal L=0,
\quad
\nabla_{\theta_p}\mathcal L\ne0,
\quad
\nabla_{\theta_{l,0}}\mathcal L\ne0,
\quad
\nabla_{\theta_{l,7}}\mathcal L\ne0.
$$
{% endraw %}

中间六层、token embedding、final norm 和 tied LM Head 仍冻结。SFT LR 降到 `5e-6`，因为更新的是已有语言能力的参数，大步长更容易造成 catastrophic forgetting。

### 6.6 为什么标准路线先 Pretrain 再 SFT

```text
llm_768.pth
  → VLM PT：只移动视觉接口
  → pretrain_vlm_768.pth
  → VLM SFT：联合调整接口、首层、末层
  → sft_vlm_768.pth
```

先对齐再指令微调，把“视觉特征完全错位”和“指令行为尚未形成”两个问题拆开，降低 SFT 初期的联合优化难度。

### 6.7 本次真实训练跳过 Pretrain 的含义

远端实际参数：

```text
--from_weight llm
--freeze_llm 1
--data_path sft_i2t.parquet
--max_seq_len 768
--learning_rate 5e-6
```

实际路线是 `llm_768.pth → sft_vlm_768.pth`。这在代码上合法，因为 SFT 数据包含 caption；但它不与两阶段训练严格等价：

- Projector 从随机初始化开始，却使用较小的 SFT LR；
- Block 0/7 从第一步就接收尚未对齐的视觉向量；
- caption 与问答 loss 同时竞争 Projector 表示空间；
- 是否达到相同质量，必须做对照实验。

本次结果只能证明“直接 SFT 路线能收敛并保存有效 checkpoint”，不能证明“跳过 Pretrain 与标准路线性能相同”。

## 7. 真实图片的 Token 轨迹与特征值

固定样本来自 `sft_i2t.parquet` 的 `row_group=0,row=0`，问题为：

```text
How does the use of negative space and color contrast
contribute to the overall impact of the image?
```

数据集中的真实标准回答为：

```text
The use of negative space, which is the white and unoccupied area
surrounding the figure, emphasizes the subject and draws the viewer's
attention directly to the character. This technique creates a stark
contrast with the golden armor, making the figure the focal point of
the composition. The limited use of color, primarily gold against the
white background, adds a dramatic and regal quality to the image. It
suggests that the character is perhaps a solitary and significant figure
within the context of its environment. The high contrast between the
character and the background also gives the impression of radiance and
power, potentially reflective of the character's status or abilities
within the game's narrative.
```

加载最终 `sft_vlm_768.pth` 后，用同一图片与问题做 greedy decoding，模型实际回答为：

```text
The use of negative space and color contrast in the image contributes
to the overall impact of the image by creating a sense of depth and
dimension. The use of negative space creates a sense of depth and
dimension, which can be interpreted as a visual metaphor for the human
form and the unknown. The color contrast, which is often associated with
the dark, orange, and yellow hues, contributes to the overall sense of
depth and dimension. The use of color also creates a sense of depth and
dimension, which can be interpreted as a visual metaphor for the human
form and the unknown. The contrast between the negative space and the
color of the image creates a sense of depth and dim
```

这是原始解码结果，不是人工润色版本。对照可以看到：

| 检查项 | 标准回答 | 模型实际回答 |
|---|---|---|
| negative space | 强调主体、引导注意力 | 提到空间关系，但反复描述 depth/dimension |
| color contrast | 白色背景与金色盔甲、dramatic/regal | 错误扩展为 dark/orange/yellow |
| 构图结论 | focal point、radiance、power | human form、unknown，语义偏移 |
| 完整性 | 完整回答并包含 EOS | 192 token 达到上限，末尾截断在 `dim` |
| 重复问题 | 无明显重复 | `sense of depth and dimension` 多次重复 |

因此这个验证例子同时包含正向证据和失败证据：模型确实使用了图片条件并回答了正确主题，但细节准确性、重复控制与 EOS 学习仍明显不足。后面的 teacher-forcing loss 衡量“给定标准前缀时预测下一个 token”的能力，不能替代自由生成质量评价。

真实 shape：

```text
pixel_values             [1,3,256,256]
SigLIP last hidden       [1,64,768]
Projector output         [1,64,768]
input_ids / labels       [1,768]
text / fused embeddings  [1,768,768]
final hidden states      [1,768,768]
logits                    [1,768,6400]
```

视觉区间替换前后 L2 改变量为 `78.5579528809`；position 68 之后的最大绝对变化为 `0.0`，验证了代码只替换 positions `4..67`。

![最终 SFT 权重下真实 SigLIP2 与 Projector Token 特征地图](/images/minimind-v-real-training/06-real-token-feature-map.png)

| 特征 | min | mean | max |
|---|---:|---:|---:|
| SigLIP2 token L2 | 11.6100 | 42.8321 | 50.7136 |
| Projector token L2 | 4.8966 | 9.7365 | 11.2256 |
| 同位置 cosine | -0.0710 | 0.01657 | 0.06727 |

Token 0 前八维：

```text
SigLIP    [ 0.0809, 0.5630, 0.7564, 0.1409, 1.4431, 0.4582,-1.5615, 0.5609]
Projector [-0.1908, 0.6970, 0.4997,-0.1685, 0.1530, 0.4171,-0.3179,-0.0503]
```

平均 cosine 接近 0 不代表信息消失。Projector 含 LayerNorm、两次 Linear 与 GELU，本来就不是恒等映射；这里也不是注意力强度或语义相似度。

## 8. 从 logits 到真实 loss

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```

```text
logits        [1,768,6400]
shift_logits  [1,767,6400]
shift_labels  [1,767]
```

assistant 标签位于原序列 `99..287`，共有 189 个有效 target，其余 578 个 shifted label 为 `-100`。

![真实 label mask、Token 概率与最终 Cross-Entropy](/images/minimind-v-real-training/05-loss-from-mask-to-scalar.png)

{% raw %}
$$
p_t=\frac{\exp z_{t-1,y_t}}
{\sum_{v=1}^{6400}\exp z_{t-1,v}},
\qquad \ell_t=-\log p_t.
$$
{% endraw %}

| target | 目标概率 | 单 token CE | top-1 |
|---|---:|---:|---|
| `The` | 0.846626 | 0.166496 | `The` |
| ` use` | 0.874382 | 0.134238 | ` use` |
| ` of` | 0.999338 | 0.000662 | ` of` |
| `,` | 0.026007 | 3.649381 | ` and` |
| ` wh` | 0.000599 | 7.419586 | ` cent` |
| ` rad` | 0.0000649 | 9.642717 | ` a` |

{% raw %}
$$
\sum_{t\in\mathcal A}\ell_t=321.6816711426,
\qquad
\bar{\mathcal L}=\frac{321.6816711426}{189}=1.7020194530.
$$
{% endraw %}

```text
model loss = 1.7020195723
manual CE  = 1.7020194530
abs diff   = 1.1920928955e-7
```

浮点误差来自归约顺序。该结果同时验证 shift、assistant mask、词表维度与 `ignore_index=-100`。

## 9. 推理时视觉编码器为什么只运行一次

生成第一步 `start_pos=0`，执行视觉编码、Projector、embedding replacement 并建立 KV cache。后续每步 `start_pos` 大于 0，跳过视觉分支：

```python
if pixel_values is not None and start_pos == 0:
    vision_tensors = vision_proj(get_image_embeddings(...))
    hidden_states = count_vision_proj(...)
```

真实观测：prompt 105 token、生成 192 token、视觉编码器调用 1 次、CPU 2.312 秒。模型能围绕 negative space 和 color contrast 生成相关文本，但在上限前未输出 EOS 并出现重复。loss 可复算和链路有效，不等于生成质量已充分解决。

## 10. 结论

1. MiniMind 文本 Pretrain 与 SFT 使用同一网络和 CE；Pretrain 监督全部有效 token，SFT 只监督 assistant token。
2. MiniMind-V 在 MiniMind 前增加 frozen SigLIP2 与 Projector，用 64 个 768 维视觉向量替换序列中的 64 个 embedding 槽位。
3. VLM Pretrain 与 SFT 也共享 forward 和 assistant-only CE；差异在数据、初始化、$L$、LR 和梯度边界。
4. 本次训练跳过 VLM Pretrain，只证明直接 SFT 路线能运行与收敛，不证明它与标准两阶段路线等价。

从优化角度看，VLM Pretrain 是“固定语言空间，移动视觉接口”；VLM SFT 是“保持视觉编码器冻结，让视觉接口与语言主干边界共同适应图像指令”。

## 附录：证据边界与源码索引

| 项目 | 真实值 |
|---|---|
| MiniMind / MiniMind-V commit | `512eed0` / `740d467` |
| SFT 数据 | 2,904,511 行，4,934,887,104 bytes |
| 实际训练 | `batch=4, L=768, freeze_llm=1, from_weight=llm` |
| 最终 step / 最后 batch loss | `726128/726128` / `2.1096` |
| 固定样本 loss | `1.7020195723` |

核心源码：

```text
MiniMind:   model/model_minimind.py, dataset/lm_dataset.py,
            trainer/train_pretrain.py, train_full_sft.py,
            train_dpo.py, train_ppo.py, train_grpo.py
MiniMind-V: model/model_vlm.py, dataset/lm_dataset.py,
            trainer/trainer_utils.py,
            train_pretrain_vlm.py, train_sft_vlm.py
```

本文的源码、真实数值与图表已经完成本地 Hexo 构建校验后发布。
