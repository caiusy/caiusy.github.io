---
title: MiniMind PPO 源码拆解：一条 JSONL 如何走到 GAE 与 Clipped Loss
date: 2026-08-01 23:19:00
updated: 2026-08-01 23:19:00
mathjax: true
description: >-
  从 MiniMind 的真实 RLAIFDataset 样本出发，逐步追踪 prompt、rollout、old log probabilities、Actor-Critic、terminal reward、TD residual、GAE、returns，以及 PPO clipped actor/critic loss 的来源、维度与代码位置。
categories:
  - AI与大模型
  - 深度学习
tags:
  - MiniMind
  - PPO
  - RLHF
  - Actor-Critic
  - GAE
  - RLAIF
  - LLM-Alignment
type: deep-dive
difficulty: advanced
review_status: published
cover: /images/minimind-ppo-source-code/01-ppo-end-to-end.svg
---

PPO 最难的地方，往往不是把公式背下来，而是回答这些看似简单的问题：

`prompt` 到底包含什么？`output_ids` 和 `completion_ids` 有什么区别？`old_logp` 从哪来？Critic 的 label 是 advantage 还是 returns？终局只有一个 reward，为什么最后每个 token 都能获得学习信号？

这篇文章不从抽象强化学习符号开始，而是沿着 MiniMind 的真实 PPO 代码走一遍：拿一条 `rlaif.jsonl` 样本，追踪它如何变成 `[1,561]` 的 prompt，如何生成 `[1,106]` 的 completion，最后如何汇入一个标量 total loss。

<!-- more -->

![MiniMind PPO 从 JSONL、rollout、reward、GAE 到 total loss 的完整数据流](/images/minimind-ppo-source-code/01-ppo-end-to-end.svg)

## 先给出全局结论

MiniMind PPO 的主链可以压缩为一句话：

> Actor 对完整 prompt 在线生成 completion，并保存每个生成 token 的 old log probability；Reward 给整条回答打分，Critic 估计每个生成状态的未来回报；TD 与 GAE 将终局结果分配给每个 token，得到教 Actor 的 advantages 和教 Critic 的 returns；最后用 policy clip、Reference KL 与 value clip 限制更新幅度。

本文使用的真实 rollout 维度是：

```text
B = 1      batch size
P = 561    padding 后的 prompt 宽度
R = 106    completion 宽度
L = 667    完整序列宽度，L = P + R
H = 768    hidden size
V = 6400   vocabulary size
```

主张量链如下：

```text
prompt_ids                    [1, 561]
output_ids                    [1, 667]
completion_ids                [1, 106]
old_logps / old_values        [1, 106]
token_rewards                 [1, 106]
advantages / returns          [1, 106]
actor/value token losses      [1, 106]
total loss                    []
```

注意：`P` 与 `R` 是 batch Tensor 的宽度。batch 大于 1 时，每条样本的真实有效长度可以不同，padding 后才共享同一个宽度。

## 1. 数据集：为什么是 `conversations[:-1]`

MiniMind 的 `RLAIFDataset` 读取 JSONL：

```python
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024,
                 thinking_ratio=0.5):
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset(
            "json", data_files=jsonl_path, split="train"
        )

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True,
        )
```

真实第 0 条样本共有 6 条消息，末尾是：

```text
assistant：这些智能家居产品需要哪些前提条件才能够使用？
user：请回答这个问题。
assistant：""
```

最后一条空 assistant 只是占位符。`conversations[:-1]` 删除它，再通过 `add_generation_prompt=True` 把生成起点放到 prompt 尾部。

这一步决定了训练范式：

- 如果直接把现成 assistant 答案当监督标签，接近 SFT。
- PPO 要让当前 Actor 在线生成新回答，再根据 reward 更新策略。

### Prompt 不是最后一句用户问题

这里的 prompt 是前 5 条历史消息经 chat template 渲染后的完整字符串，包括多轮 user/assistant 对话、角色标记，以及新的 assistant 生成起点。

固定随机种子 42、`thinking_ratio=0.9` 时，真实结果是：

```text
prompt 字符数          991
input_ids              [1, 561]
attention_mask         [1, 561]
有效 prompt token      561
```

因此，“prompt 是什么”与“最后一个 user message 是什么”是两个问题。最后一句用户消息只是完整 prompt 的最后一部分。

### 数据增强发生在哪里

这条数据路径有两类随机变化：

1. 首条消息不是 system 时，预处理约有 20% 概率补一个中英文 system prompt。
2. `thinking_ratio` 控制是否使用 open-thinking generation prompt；本文配置为 0.9。

源码中虽然还有 `post_processing_chat()`，但它没有被这条 `RLAIFDataset` 路径调用，不能把其中的规则算到 PPO 数据增强里。

### Batch padding 的维度

真实样本 0 和 1 的 prompt 长度分别是 561 与 470。组成 `B=2` 的 batch 并左填充后：

```text
input_ids              [2, 561]
attention_mask         [2, 561]
sample 0               [真实 token × 561]
sample 1               [PAD × 91] [真实 token × 470]
```

`input_ids` 存 token ID，`attention_mask` 用 1 标记有效 token、0 标记 padding。

## 2. Actor、Critic、Reward、Reference 到底是谁

PPO 同时出现多个模型，很容易把职责混在一起。

![PPO 中 Actor、Critic、Reward Model 与 Reference Model 的角色分工](/images/minimind-ppo-source-code/02-actor-critic-roles.svg)

### Actor：负责选择下一个 token

Actor 就是正在训练的因果语言模型策略。给定当前前缀状态，它输出词表 logits，再经 softmax 得到下一个 token 的条件概率。

{% raw %}
$$
\pi_{\theta}(a_t\mid s_t)
=
P_{\theta}(\text{next token}=a_t\mid\text{prefix}=s_t)
$$
{% endraw %}

MiniMind 的词表大小为 6400，因此完整 Actor logits 为：

```text
hidden states           [B, L, 768]
Actor logits            [B, L, 6400]
```

### Critic：不选 token，只估计未来回报

MiniMind 的 Critic 继承同类 Transformer 主干，并新增一个 value head：

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```

继承来的 LM head 没有被物理删除，但 Critic forward 使用新增的 `value_head`，每个位置只输出一个 value：

```text
hidden states           [B, L, 768]
value_head output       [B, L, 1]
squeeze                 [B, L]
response values         [B, R]
```

### Reward Model：回答完成后给整条轨迹评分

Reward Model 与 Critic 不是同一个模型。

- Reward 在回答结束后输出整条回答的分数 `rewards [B]`。
- Critic 在每个生成状态输出价值预测 `values [B,R]`。

MiniMind 还叠加了长度、thinking 格式与重复惩罚。若回答包含 `</think>`，规则奖励检查较完整的 response，而 Reward Model 只接收 prompt 加 `</think>` 后的 answer。两者的评分对象并不完全相同。

### Reference Model：防止策略为追 reward 跑偏

Reference 是冻结的稳定策略，输出 `ref_logps [B,R]`，用于 KL penalty。它不是 old policy。

本实现没有长期维护另一套 old Actor。所谓 old policy，是 rollout 时保存下来的 `old_logps`；Reference 则是独立的冻结模型。

## 3. Rollout：prompt、output、completion 如何对齐

Torch rollout 的核心操作是：

```python
output_ids = model.generate(
    input_ids=prompt_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=0.8,
)

prompt_len = prompt_ids.size(1)
completion_ids = output_ids[:, prompt_len:]
```

真实结果：

```text
prompt_ids             [1, 561]
output_ids             [1, 667]
completion_ids         [1, 106]
old_logps              [1, 106]
completion_mask        [1, 106]
EOS                    completion[105]
```

最重要的关系不是 Tensor 数值相加，而是沿序列维拼接：

```text
output_ids = concat(prompt_ids, completion_ids, dim=sequence)
```

### old logp 是怎样得到的

每个生成位置原本有 6400 个候选概率。代码对完整 `output_ids` 再做 Actor forward，然后只 gather 实际生成 token 的 log probability：

```python
logits = model(input_ids).logits[:, :-1, :]
log_probs = logits.log_softmax(dim=-1)
selected_logps = gather(log_probs, generated_token_ids)
```

维度变化：

```text
response logits         [B, R, V]
generated token ids     [B, R]
selected old logps      [B, R]
```

对数概率越接近 0，原概率越大；越负，原概率越小。使用 log probability 可以把概率乘除转成加减，并提高数值稳定性。

### 为什么位置要右移一位

自回归模型的 `logits[t]` 预测 `token[t+1]`。

真实样本中：

```text
completion 位于完整序列       561 ... 666
用于预测它的 logits/value     560 ... 665
```

所以：

```text
logits[560] 预测 completion[0]
value[560]  表示生成 completion[0] 前的状态价值
```

本样本最后生成 token 恰好是 EOS，因此最后一个选中 logits 预测 EOS；如果生成因达到长度上限而停止，这条结论并不成立。

### response mask 是否包含 EOS

最终 policy/value mask 保留到第一个 EOS 为止，并且包含这个 EOS；EOS 之后与 padding 位置为 0。

Torch rollout 当前先返回全 1 的 `completion_mask`，PPO 主循环随后结合 EOS 和 padding 构造真正用于 GAE 与 loss 的 `resp_policy_mask` / `resp_value_mask`。

### 一个值得保留的实现差异

`generate()` 使用 `temperature=0.8` 采样，但保存 old logp 时对未温度化 logits 做 `log_softmax`。严格来说，保存的 old logp 不是温度化采样分布的精确 log probability，后续 ratio 也就不是生成分布与 current policy 的完全精确 importance ratio。

这不影响我们读懂数据流，但在讨论“严格 on-policy”语义时不能忽略。

## 4. Reward：一个分数如何放进 106 个 token

MiniMind 的完整终局 reward 为：

{% raw %}
$$
R_{\mathrm{final}}
=
R_{\mathrm{length}}
+R_{\mathrm{think}}
+R_{\mathrm{format}}
-R_{\mathrm{repeat}}
+R_{\mathrm{RM}}
$$
{% endraw %}

本文真实 rollout 的规则部分：

```text
回答长度合格           +0.50
thinking 长度合格      +1.00
think 格式合格         +0.25
重复惩罚               -0.00
规则奖励合计            1.75
```

再加上 Reward Model score，得到 `rewards [B]`。

但 GAE 工作在逐 token 维度 `[B,R]`。代码先创建全 0 Tensor，再把整条 reward 放到最后有效 token：

```python
token_rewards = torch.zeros_like(old_resp_logp)
last_idx = resp_lengths - 1
token_rewards[batch_index, last_idx] += rewards
```

因此真实 shape 是：

```text
rewards                 [1]
token_rewards           [1, 106]
token_rewards           [0, 0, ..., 0, final_reward]
```

这一步必须记牢：`token_rewards` 不是 GAE 的输出，而是 GAE 的输入。

## 5. TD residual：Critic 这一步到底估错了多少

Bellman 一步目标是“当前 reward 加下一状态的折扣价值”：

{% raw %}
$$
\mathrm{TDTarget}_t
=
r_t+\gamma V_{\mathrm{old}}(s_{t+1})
$$
{% endraw %}

用它减去 Critic 原预测，得到 TD residual：

{% raw %}
$$
\delta_t
=
r_t
+\gamma V_{\mathrm{old}}(s_{t+1})
-V_{\mathrm{old}}(s_t)
$$
{% endraw %}

各项来源：

| 变量 | 含义 | 来源 |
|---|---|---|
| $r_t$ | 当前 token reward | `token_rewards [B,R]` |
| $V_{\mathrm{old}}(s_t)$ | 当前状态旧价值预测 | old Critic forward |
| $V_{\mathrm{old}}(s_{t+1})$ | 下一状态旧价值预测 | 同一次 forward 的下一位置 |
| $\gamma$ | 未来折扣 | 本项目为 1.0 |
| $\delta_t$ | 一步目标与旧预测的差 | GAE 的原料 |

直觉上：

- `delta` 为正：看到下一状态后，发现 Critic 原来低估了当前状态。
- `delta` 为负：Critic 原来高估了当前状态。
- 终止位置没有下一状态，本实现令 next value 为 0。

TD residual 来自 Bellman/时序差分框架，不是 PPO 发明的。

## 6. GAE：把终局结果传给更早 token

语言模型通常到回答结束才拿到 reward，但前面每个 token 都参与了这次结果。GAE 的作用就是信用分配。

![三 token 例子中 TD residual、GAE advantage 与 returns 的反向递推](/images/minimind-ppo-source-code/03-gae-credit-assignment.svg)

GAE 的递推形式：

{% raw %}
$$
A_t
=
\delta_t
+\gamma\lambda A_{t+1}
$$
{% endraw %}

展开后：

{% raw %}
$$
A_t
=
\delta_t
+(\gamma\lambda)\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}
+\cdots
$$
{% endraw %}

`lambda` 控制未来 TD residual 向前传播多远：

- `lambda=0`：只看当前一步，更依赖 Critic，方差低但偏差可能更高。
- `lambda` 接近 1：终局信息传得更远，更接近完整回报，但方差可能更高。
- MiniMind 使用 `lambda=0.95`。

### 三 token 手算

假设：

```text
token_rewards = [0.0, 0.0, 1.0]
old_values    = [0.2, 0.3, 0.4]
gamma         = 1.0
lambda        = 0.95
```

从后向前：

{% raw %}
$$
\delta_2=1.0-0.4=0.6,
\qquad
A_2=0.6
$$
{% endraw %}

{% raw %}
$$
\delta_1=0+0.4-0.3=0.1,
\qquad
A_1=0.1+0.95\times0.6=0.67
$$
{% endraw %}

{% raw %}
$$
\delta_0=0+0.3-0.2=0.1,
\qquad
A_0=0.1+0.95\times0.67=0.7365
$$
{% endraw %}

所以：

```text
raw_advantages = [0.7365, 0.6700, 0.6000]
```

Returns 由 raw advantages 与 old values 构造：

{% raw %}
$$
\mathrm{Return}_t
=
A_t^{\mathrm{raw}}
+V_{\mathrm{old},t}
$$
{% endraw %}

```text
returns = [0.9365, 0.9700, 1.0000]
```

现在可以精确回答“label 是什么”：

```text
raw advantages          用来构造 returns
normalized advantages   Actor loss 的逐 token 权重
returns                 Critic 的训练目标，也就是 Critic label
current values          Critic 当前预测
```

代码先计算 `returns = raw_advantages + old_values`，之后才标准化 advantages，因此 returns 不会受到 advantage 标准化影响。

### 从后向前会不会很慢

不会。Critic 先对完整 `[B,P+R]` 序列做一次并行 forward，得到所有位置的 value。GAE 只是对已有 `[B,R]` Tensor 做一次复杂度为 `O(BR)` 的反向循环。

它既不是逐 token 重跑 Critic，也不是神经网络的 `loss.backward()`。PPO 的主要计算仍在自回归生成和 Transformer 前向/反向。

## 7. PPO Loss：三道安全闸

PPO 的 loss 不是单独一个公式。MiniMind 的核心组成是 Actor clipped loss、Reference KL penalty 和 Critic clipped value loss；启用 MoE 时还会增加 auxiliary loss。

![Actor clipped loss、Reference KL 和 Critic value loss 的输入、系数与约束对象](/images/minimind-ppo-source-code/04-ppo-losses.svg)

### 7.1 Actor clipped loss

当前策略与 old policy 对同一生成 token 的概率比：

{% raw %}
$$
\mathrm{ratio}_t
=
\exp\left(
\log p_{\mathrm{current},t}
-\log p_{\mathrm{old},t}
\right)
$$
{% endraw %}

注意：ratio 是 `[B,R]`，每个有效 token 一个，不是整条回答只有一个。

本项目 `clip_epsilon=0.2`：

{% raw %}
$$
\mathrm{ratio}^{\mathrm{clip}}_t
=
\operatorname{clip}(\mathrm{ratio}_t,0.8,1.2)
$$
{% endraw %}

逐 token Actor loss：

{% raw %}
$$
L_{\mathrm{actor},t}
=
\max\left(
-A_t\,\mathrm{ratio}_t,
-A_t\,\mathrm{ratio}^{\mathrm{clip}}_t
\right)
$$
{% endraw %}

正 advantage 会推动 token 概率上升，负 advantage 会推动它下降；clip 让一次更新不要走太远。

### 7.2 Reference KL penalty

令：

{% raw %}
$$
d_t
=
\log p_{\mathrm{ref},t}
-\log p_{\mathrm{current},t}
$$
{% endraw %}

MiniMind 使用的逐 token KL 形式为：

{% raw %}
$$
L_{\mathrm{KL}}
=
\operatorname{masked\ mean}_t
\left(
\exp(d_t)-d_t-1
\right)
$$
{% endraw %}

`kl_coef=0.02`。它不直接判断答案好坏，而是约束 Actor 不要偏离冻结 Reference 太远。

### 7.3 Critic clipped value loss

Critic 的目标是 returns。先计算当前 value 的普通平方误差：

{% raw %}
$$
E_{1,t}
=
\left(
V_{\mathrm{current},t}
-\mathrm{Return}_t
\right)^2
$$
{% endraw %}

再把当前 value 限制在 old value 附近：

{% raw %}
$$
V_{\mathrm{clip},t}
=
\operatorname{clip}
\left(
V_{\mathrm{current},t},
V_{\mathrm{old},t}-0.2,
V_{\mathrm{old},t}+0.2
\right)
$$
{% endraw %}

{% raw %}
$$
E_{2,t}
=
\left(
V_{\mathrm{clip},t}
-\mathrm{Return}_t
\right)^2
$$
{% endraw %}

逐 token value loss：

{% raw %}
$$
L_{\mathrm{value},t}
=
\frac{1}{2}
\max(E_{1,t},E_{2,t})
$$
{% endraw %}

它一边拟合 returns，一边惩罚 value 相对 old prediction 一次跳太远。

### 7.4 Total loss

远程实际配置：

```text
clip_epsilon       0.2
cliprange_value    0.2
kl_coef            0.02
vf_coef            0.5
use_moe            0
```

{% raw %}
$$
L_{\mathrm{total}}
=
L_{\mathrm{actor}}
+0.02L_{\mathrm{KL}}
+0.5L_{\mathrm{value}}
+L_{\mathrm{MoE\ aux}}
$$
{% endraw %}

所有 token loss 都先乘 response mask，再除以有效 token 数，从 `[B,R]` 聚合成标量 `[]`，最后执行 `backward()`。

`reward`、`token_rewards`、`advantages` 和 `returns` 是评分、输入、权重或目标，不是额外 loss。`approx_kl` 与 `clipfrac` 是监控/early-stop 指标，也不是 loss。

## 8. 一张变量总账

| 变量 | 来源 | Shape | 是否有梯度 | 用途 |
|---|---|---:|---|---|
| `prompts` | RLAIFDataset | `list[str]` | 否 | tokenizer 输入 |
| `input_ids` | tokenizer | `[B,P]` | 否 | rollout 输入 |
| `output_ids` | Actor generate | `[B,P+R]` | 否 | 完整轨迹 |
| `completion_ids` | output 切片 | `[B,R]` | 否 | 新生成 token |
| `old_logps` | rollout Actor | `[B,R]` | 否 | ratio 基准 |
| `response_mask` | EOS + padding | `[B,R]` | 否 | 过滤无效 token |
| `rewards` | 规则 + RM | `[B]` | 否 | 整条回答评分 |
| `old_values` | rollout Critic | `[B,R]` | 否 | GAE 与 value clip |
| `token_rewards` | terminal scatter | `[B,R]` | 否 | GAE 输入 |
| `advantages` | GAE + 标准化 | `[B,R]` | 否 | Actor loss 权重 |
| `returns` | raw advantage + old value | `[B,R]` | 否 | Critic label |
| `current_logps` | current Actor | `[B,R]` | 是 | Actor loss |
| `current_values` | current Critic | `[B,R]` | 是 | value loss |
| `policy_loss` | Actor clip + KL | `[]` | 是 | 更新 Actor |
| `value_loss` | clipped value error | `[]` | 是 | 更新 Critic |
| `total_loss` | 所有项加权 | `[]` | 是 | backward |

## 9. 最容易混淆的十件事

| 误解 | 正确结论 |
|---|---|
| Prompt 只是最后一句用户问题 | Prompt 是 chat template 渲染后的完整多轮历史 |
| output 与 completion 相同 | output 是 prompt + completion；completion 只是新生成部分 |
| Actor 是另一个神秘网络 | Actor 就是负责生成 token 的因果语言模型 |
| Critic 与 Reward Model 相同 | Reward 评整条；Critic 逐状态估未来回报 |
| old policy 就是 Reference | old policy 由 rollout old logps 表示；Reference 是独立冻结模型 |
| token_rewards 是 GAE 输出 | token_rewards 在 GAE 之前构造，是 GAE 输入 |
| advantages 是 Critic label | 标准化 advantages 教 Actor；returns 教 Critic |
| 代码变量 `labels` 是 Critic label | 它是 gather logp 用的右移 token IDs |
| ratio 每条回答一个 | ratio 是 `[B,R]`，每 token 一个 |
| GAE 倒推会逐 token 跑 Critic | Critic 一次并行 forward；GAE 只是 Tensor 递推 |

## 10. 推荐的源码阅读顺序

不要从训练脚本第一行机械读到底。按对象追踪：

1. `dataset/lm_dataset.py:195-224`：`RLAIFDataset`、`conversations[:-1]`、chat template。
2. `trainer/rollout_engine.py:39-47`：先看 `RolloutResult` 输入输出合同。
3. `trainer/rollout_engine.py:71-92`：`generate()`、completion 切片与 decode。
4. `trainer/rollout_engine.py:24-36`：逐 token old logp 的 gather。
5. `trainer/train_ppo.py:84-101`：tokenizer、rollout 与 reward 接入。
6. `trainer/train_ppo.py:117-128`：右移、位置对齐、EOS 与 response mask。
7. `trainer/train_ppo.py:36-49,130-151`：Critic、token reward、GAE、returns。
8. `trainer/train_ppo.py:171-203`：current Actor/Critic 与三类 loss。
9. `trainer/train_ppo.py:208-214`：total loss 与 backward。

第一次调试建议先使用只读数据流脚本，观察对象与 shape，不要直接进入包含 `optimizer.step()` 和 checkpoint 保存的完整训练入口。

## 最后总结

PPO 可以看成两个学习问题共享同一条 rollout：

- Actor 学“哪些 token 决策值得增加或降低概率”，权重是 normalized advantages。
- Critic 学“当前状态未来能拿多少回报”，目标是 returns。

终局 reward 先放到最后有效 token，TD residual 衡量 Critic 的一步预测误差，GAE 再把后续误差按 `gamma × lambda` 衰减传回更早位置。

最终，Actor clip 限制策略突变，Reference KL 防止语言行为跑偏，Critic value clip 限制估值突变。所有逐 token 量保持 `[B,R]`，直到 response mask 聚合后才变成一个可执行 `backward()` 的标量 loss。

如果只记一条链，请记住：

```text
JSONL → prompt → rollout → old logps / old values
      → terminal reward → token_rewards
      → TD residual → GAE
      → advantages 教 Actor
      → returns 教 Critic
      → clipped losses → total loss
```
