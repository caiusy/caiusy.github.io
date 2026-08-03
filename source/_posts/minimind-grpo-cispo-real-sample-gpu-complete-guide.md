---
title: MiniMind GRPO/CISPO 完整拆解：真实四候选、Reward、Advantage、Loss 与 GPU 全流程
date: 2026-08-03 21:35:00
updated: 2026-08-03 21:35:00
mathjax: true
description: >-
  基于远程 WSL 与真实 MiniMind GRPO checkpoint，从同一 561-token prompt 生成 4 条真实回答，逐项计算 reward、组均值、总体标准差、advantage、Reference KL、GRPO/CISPO loss，并解释策略更新时间线与 GPU 路径。
categories:
  - AI与大模型
  - 深度学习
tags:
  - MiniMind
  - GRPO
  - CISPO
  - RLHF
  - Group-Relative-Policy-Optimization
  - Reference-Model
  - GPU
  - LLM-Alignment
type: deep-dive
difficulty: advanced
review_status: published
cover: /images/minimind-grpo-cispo-real-complete/01-grpo-full-pipeline.svg
---

这是一篇不省略中间步骤的 GRPO/CISPO 长文：从同一条真实输入生成四条候选，保留每条reward、组均值、总体标准差、advantage、逐 token loss、Reference、Old/Current/New policy与 GPU 路径的全部细节。正文完整收录原始笔记，只增加发布导航与配图。

<!-- more -->

<!-- ORIGINAL_NOTE_BEGIN: source content preserved; publication additions are image lines and MathJax wrappers -->
# MiniMind GRPO/CISPO：真实输入、四条 Rollout、Reward、Advantage、Loss 与 GPU 全流程

> 本文只讲 GRPO 框架及项目实际 CISPO loss。所有具体值来自远程 WSL `wsl-dev`、MiniMind commit `4497610`、固定 `sample_index=0, seed=42, G=4` 的真实只读 forward。没有执行 backward、optimizer.step 或保存权重。

## 阅读目标

本文从一条真实文字输入开始，展示实际 prompt、token ID 与 shape；随后让同一输入生成 4 条真实回答，逐条计算规则奖励、Reward Model score、组均值、总体标准差、advantage、逐 token GRPO/CISPO loss和最终 scalar。

## 1. 真实环境与证据边界

### 1.1 远程环境

| 项目 | 真实值 |
|---|---|
| 远程主机 | `wsl-dev` / `caius-PC` |
| MiniMind 路径 | `/home/caius/minimind` |
| MiniMind commit | `4497610` |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GPU 总显存 | 24,564 MiB |
| 驱动 | 560.94 |
| PyTorch | `2.6.0+cu124` |
| PyTorch CUDA | 12.4 |
| RLAIF 数据 | `/home/caius/datasets/minimind_dataset/rlaif.jsonl` |
| 数据行数 | 19,502 |
| Reward Model | `/home/caius/internlm2-1_8b-reward` |
| PPO 权重 | `out/lifeos_agent_ppo_v1_768.pth` |
| GRPO 权重 | `out/lifeos_agent_grpo_v1_768.pth` |
| Reference 起点 | `out/lifeos_agent_dpo_v1_768.pth` |

### 1.2 三类数字必须分开

本文会遇到三类数字：

1. **真实数据值**：JSONL 内容、prompt 长度、真实生成文本。
2. **真实 forward 值**：logp、value、Reward Model score、advantage、KL、loss、PyTorch 显存。
3. **历史训练日志值**：训练 step、平均 reward、平均长度、耗时和最终 checkpoint。

真实 forward 是“固定 checkpoint 在当前环境中的一次可复算观察”，不等同于历史训练中的某一步；历史日志只有聚合指标，不能从中反推出当时每个 token 的 logp。

## 2. 符号、shape 与类型总表

| 符号 | 含义 | 本文真实值 |
|---|---|---:|
| `B` | 原始 prompt batch size | 1 |
| `G` | 每个 prompt 的生成数 | PPO 为 1；历史 GRPO 为 4 |
| `P` | padding 后的 prompt tensor 宽度 | 561 |
| `R` | completion tensor 宽度 | PPO 106；GRPO 256 |
| `L` | 完整序列宽度，`P+R` | PPO 667；GRPO 817 |
| `H` | hidden size | 768 |
| `V` | vocabulary size | 6,400 |
| `i` | rollout / 候选回答索引 | `0...B×G-1` |
| `t` | completion token 位置 | `0...R-1` |
| `m_{i,t}` | completion 有效位置 mask | 0 或 1 |
| `R_i` | 第 i 条完整回答的终局 reward | scalar |
| `A_i` | GRPO 第 i 条回答的组相对 advantage | scalar |
| `A_{i,t}` | PPO 第 i 条回答第 t 个位置的 GAE advantage | scalar |

需要特别区分：

- `R` 在 shape 中表示 response width，不表示 reward。
- `R_i` 才表示第 i 条回答的 reward。
- tensor width 是 padding 后的统一宽度，不一定等于每条回答的有效 token 数。

## 3. 贯穿全文的真实样本
![GRPO/CISPO 真实四候选完整链路](/images/minimind-grpo-cispo-real-complete/01-grpo-full-pipeline.svg)


远程 `rlaif.jsonl` 第 0 行共有 6 条消息：

```text
user：给出张明（销售经理）和刘琳（智能家居研发工程师）的角色信息。

assistant：两人进行长对话，其中明确提到：
1. 使用前需要稳定 Wi-Fi；
2. 提供详细说明书，包含安装和使用步骤；
3. 有数据加密、防火墙、实时安全更新；
4. 有在线客服和售后服务。

user：基于以上对话提出一个问题。
assistant：这些智能家居产品需要哪些前提条件才能够使用？
user：请回答这个问题。
assistant：""
```

这道题的直接答案至少应包括：

- 客户需要稳定的 Wi-Fi 网络；
- 应阅读说明书，并按说明完成安装、连接和使用。

安全措施与售后虽然在上下文中出现，但不是“使用前提”的主要答案。

## 3.1 这条真实样本的原文、模板与 Token

这一节先不抽象。下面只追踪 `sample_index=0`、`seed=42` 这一条真实轨迹。

### 步骤 0：JSONL 中原始输入是什么

原始 `conversations` 的 6 条消息如下。第二条 assistant 很长，但它不是标签，而是当前问题所依赖的历史上下文：

```json
[
  {
    "role": "user",
    "content": "基于以下角色信息完成一段对话\n张明（30岁），一位IT公司的销售经理。外表干练，擅长沟通。最近他需要协助生产部门促销一批新的智能家居产品，但是他对于这些新产品的技术方面并不够熟悉。\n刘琳（25岁），一名来自生产部门的研发工程师。外表看起来有些内向，但实则是一名技术精湛的工程师。最近生产的智能家居产品是她和她的团队开发的。"
  },
  {
    "role": "assistant",
    "content": "张明：嗨，刘琳，我听说智能家居产品已经进入了我们公司，我真的很想能够了解一下这些产品的技术方面，以便于在销售中向客户传递正确的信息。\n刘琳：嗨，张明，很高兴为您介绍。这些智能家居产品基于物联网技术开发，并且已经通过了一系列的测试，具有很高的品质保证，其中包括智能插座、智能门锁等产品，您想了解哪些方面？\n张明：首先，我想知道产品的使用前提是什么，例如，在产品安装之前客户需要做哪些准备，以及在使用产品时需要注意哪些事项？\n刘琳：客户使用产品之前需要保证有一个稳定的Wi-Fi网络，因为产品只能够通过Wi-Fi网络进行连接和控制。此外，我们提供了一份详细的使用说明书，其中包含了如何安装和使用产品的所有步骤。\n张明：我也想问一下产品的安全性能如何，客户的隐私是否会被泄露？\n刘琳：我们的产品具有完善的安全措施，可以保证客户的隐私不会泄露。这些措施包括数据加密技术、防火墙以及实时的安全更新等。\n张明：听起来很不错，但是我想问一下，如果客户在使用产品时遇到问题，我们应该如何解决？\n刘琳：我们提供了在线客服服务以及售后服务，如果客户在使用过程中遇到问题，可以随时和我们取得联系，我们会在第一时间为客户解决问题。\n张明：感谢您对我的解答，我对这些产品有了更全面的了解，并且可以更好地向客户介绍这些产品了。\n刘琳：没关系，很高兴能够为您提供帮助。"
  },
  {"role": "user", "content": "基于以上对话提出一个问题。"},
  {"role": "assistant", "content": "这些智能家居产品需要哪些前提条件才能够使用？"},
  {"role": "user", "content": "请回答这个问题。"},
  {"role": "assistant", "content": ""}
]
```

### 步骤 1：`[:-1]` 后剩下什么

```text
before = 6 条
after  = 5 条
删除   = {"role":"assistant","content":""}
保留角色顺序 = [user, assistant, user, assistant, user]
```

### 步骤 2：真正交给 Tokenizer 的文字是什么

固定 seed 42 后，本次 `open_thinking=True`。Chat template 输出 991 个字符。下面是实际 prompt，不是示意文本：

```text
<|im_start|>user
基于以下角色信息完成一段对话
张明（30岁），一位IT公司的销售经理。外表干练，擅长沟通。最近他需要协助生产部门促销一批新的智能家居产品，但是他对于这些新产品的技术方面并不够熟悉。
刘琳（25岁），一名来自生产部门的研发工程师。外表看起来有些内向，但实则是一名技术精湛的工程师。最近生产的智能家居产品是她和她的团队开发的。<|im_end|>
<|im_start|>assistant
<think>

</think>

张明：嗨，刘琳，我听说智能家居产品已经进入了我们公司，我真的很想能够了解一下这些产品的技术方面，以便于在销售中向客户传递正确的信息。
刘琳：嗨，张明，很高兴为您介绍。这些智能家居产品基于物联网技术开发，并且已经通过了一系列的测试，具有很高的品质保证，其中包括智能插座、智能门锁等产品，您想了解哪些方面？
张明：首先，我想知道产品的使用前提是什么，例如，在产品安装之前客户需要做哪些准备，以及在使用产品时需要注意哪些事项？
刘琳：客户使用产品之前需要保证有一个稳定的Wi-Fi网络，因为产品只能够通过Wi-Fi网络进行连接和控制。此外，我们提供了一份详细的使用说明书，其中包含了如何安装和使用产品的所有步骤。
张明：我也想问一下产品的安全性能如何，客户的隐私是否会被泄露？
刘琳：我们的产品具有完善的安全措施，可以保证客户的隐私不会泄露。这些措施包括数据加密技术、防火墙以及实时的安全更新等。
张明：听起来很不错，但是我想问一下，如果客户在使用产品时遇到问题，我们应该如何解决？
刘琳：我们提供了在线客服服务以及售后服务，如果客户在使用过程中遇到问题，可以随时和我们取得联系，我们会在第一时间为客户解决问题。
张明：感谢您对我的解答，我对这些产品有了更全面的了解，并且可以更好地向客户介绍这些产品了。
刘琳：没关系，很高兴能够为您提供帮助。<|im_end|>
<|im_start|>user
基于以上对话提出一个问题。<|im_end|>
<|im_start|>assistant
<think>

</think>

这些智能家居产品需要哪些前提条件才能够使用？<|im_end|>
<|im_start|>user
请回答这个问题。<|im_end|>
<|im_start|>assistant
<think>
```

注意最后停在 `<think>\n`，这正是 Actor 开始续写的位置。

### 步骤 3：具体切成了什么 token

Tokenizer 输出：

```text
input_ids.dtype      = torch.int64
input_ids.shape      = [1,561]
attention_mask.shape = [1,561]
valid tokens         = 561
```

开头 20 个真实 token：

| index | token id | decode 后文本 |
|---:|---:|---|
| 0 | 1 | `<|im_start|>` |
| 1 | 832 | `us` |
| 2 | 311 | `er` |
| 3 | 234 | 换行 |
| 4 | 2061 | `基于` |
| 5 | 850 | `以下` |
| 6 | 2351 | `角色` |
| 7 | 839 | `信息` |
| 8 | 1996 | `完成` |
| 9 | 3875 | `一段` |
| 10 | 3109 | `对话` |
| 11 | 234 | 换行 |
| 12 | 2370 | `张` |
| 13 | 851 | `明` |
| 14 | 555 | `（` |
| 15 | 2921 | `30` |
| 16 | 4744 | `岁` |
| 17 | 2097 | `），` |
| 18 | 5752 | `一位` |
| 19 | 6103 | `IT` |

末尾 30 个真实 token：

| index | token id | decode 后文本 |
|---:|---:|---|
| 531 | 5949 | `家居` |
| 532 | 1306 | `产品` |
| 533 | 630 | `需要` |
| 534 | 2489 | `哪些` |
| 535 | 938 | `前` |
| 536 | 454 | `提` |
| 537 | 2041 | `条件` |
| 538 | 2478 | `才` |
| 539 | 934 | `能够` |
| 540 | 533 | `使用` |
| 541 | 776 | `？` |
| 542 | 2 | `<|im_end|>` |
| 543 | 234 | 换行 |
| 544 | 1 | `<|im_start|>` |
| 545 | 832 | `us` |
| 546 | 311 | `er` |
| 547 | 234 | 换行 |
| 548 | 960 | `请` |
| 549 | 3854 | `回答` |
| 550 | 4582 | `这个问题` |
| 551 | 302 | `。` |
| 552 | 2 | `<|im_end|>` |
| 553 | 234 | 换行 |
| 554 | 1 | `<|im_start|>` |
| 555 | 1388 | `ass` |
| 556 | 570 | `ist` |
| 557 | 811 | `ant` |
| 558 | 234 | 换行 |
| 559 | 25 | `<think>` |
| 560 | 234 | 换行 |

这解释了为什么人眼看到一个单词，Tokenizer 不一定只产生一个 token。例如 `user` 被切成 `us + er`，`assistant` 被切成 `ass + ist + ant`，“前提条件”被切成 `前 + 提 + 条件`。


## 4. GRPO 真实四候选数据流

## 4.1 历史配置与当前默认值

历史 LifeOS GRPO 训练显式使用：

```text
B                 = 1
G                 = 4
max_seq_len       = 768
max_gen_len       = 256
hidden_size       = 768
num_hidden_layers = 8
learning_rate     = 3e-7
temperature       = 0.8
beta              = 0.1
loss_type         = cispo
epsilon_high      = 5.0
dtype             = bfloat16
seed              = 42
```

当前源码的 `num_generations` 默认值后来是 6。复现历史实验必须显式传 `--num_generations 4`，不能用当前默认值代替历史参数。

## 4.2 真实 shape

Rollout engine 执行：

```python
prompt_ids.repeat_interleave(G, dim=0)
```

真实张量：

```text
prompt_ids          [1,561]
重复后的 prompt      [4,561]
output_ids          [4,817]
completion_ids      [4,256]
old_logps           [4,256]
current_logps       [4,256]
reference_logps     [4,256]
completion_mask     [4,256]
rewards             [4]
grouped_rewards     [1,4]
advantages          [4]
per_token_loss      [4,256]
policy_loss         []
```

四条有效长度：

```text
[186,184,256,193]
```

第 3 条（索引 2）达到 256 token 上限，没有 EOS；其他三条的 mask 包含首个 EOS，并过滤 EOS 后的 padding。

## 4.3 四条真实回答和 reward
![四条真实回答的 Reward 与 Advantage](/images/minimind-grpo-cispo-real-complete/02-four-rollouts-reward.svg)


四条回答都把问题误解成“安全性如何”，没有直接回答稳定 Wi-Fi 和安装说明。

| 索引 | 有效 token | 回答概要 | 规则分 | RM score | final reward |
|---:|---:|---|---:|---:|---:|
| 0 | 186 | 加密、访问控制、隐私安全 | 1.75 | -0.111023 | 1.638977 |
| 1 | 184 | 数据加密、防火墙、实时更新 | 1.75 | -0.131836 | 1.618164 |
| 2 | 256 | 重复讨论加密与访问控制，未正常结束 | 1.75 | -1.102539 | 0.647461 |
| 3 | 193 | 加密、防火墙、安全更新、专业帮助 | 1.75 | -0.574219 | 1.175781 |

每条规则分 `1.75` 都是逐项算出的：

```text
完整 response 字符长度在 20–800      +0.50
thinking 内容长度在 20–300           +1.00
恰好出现一个 </think>                +0.25
answer 3-gram 重复惩罚                -0.00
规则小计                              1.75
```

Reward Model 不直接接收 MiniMind 的 `[4,256]` completion tensor。对每条回答，它把相同历史 prompt 压成一条 RM user message，把 `</think>` 后的 answer 作为 RM assistant message，再使用 InternLM2 自己的 tokenizer 编码并在末尾追加 `reward_token_id=92527`。

四条真实 RM 输入 shape：

| generation | RM `input_ids` | RM `attention_mask` | reward token | RM scalar 输出 |
|---:|---:|---:|---:|---:|
| 0 | `[1,505]` | `[1,505]` | 92527 | -0.1110229492 |
| 1 | `[1,522]` | `[1,522]` | 92527 | -0.1318359375 |
| 2 | `[1,549]` | `[1,549]` | 92527 | -1.1025390625 |
| 3 | `[1,500]` | `[1,500]` | 92527 | -0.5742187500 |

四条 RM 输入的开头 token IDs 相同：`[1,92543,1008,364,1008,334,262,70513]`。长度不同，是因为四个 answer 文本不同。

这里出现一个非常重要的事实：

> 四条语义上都没有正确回答原问题，但 reward 仍能产生清晰排名。GRPO 只能优化 reward 定义的相对优劣；reward 与任务目标错配时，算法可能稳定地优化错误方向。

## 4.4 Reward 组均值

未舍入原始值：

{% raw %}
$$
R=[1.63897705078125,1.6181640625,0.6474609375,1.17578125]
$$
{% endraw %}

组均值：

{% raw %}
$$
\mu_R=\frac1G\sum_{i=1}^{G}R_i
$$
{% endraw %}

代入：

{% raw %}
$$
\mu_R
=\frac{1.6389770508+1.6181640625+0.6474609375+1.17578125}{4}
=1.2700958252
$$
{% endraw %}

`mu` 是同一个 prompt 内的 baseline，不与其他 prompt 混合。

## 4.5 组标准差

源码使用 `unbiased=False`，即总体标准差：

{% raw %}
$$
\sigma_R
=\sqrt{\frac1G\sum_{i=1}^{G}(R_i-\mu_R)^2}
$$
{% endraw %}

真实结果：

{% raw %}
$$
\sigma_R=0.4042877257
$$
{% endraw %}

不是除以 `G-1` 的样本标准差。

## 4.6 组相对 advantage

公式：

{% raw %}
$$
A_i=\frac{R_i-\mu_R}{\sigma_R+10^{-4}}
$$
{% endraw %}

`1e-4` 防止同组 reward 几乎相同时除以 0。

逐条代入：

{% raw %}
$$
A_0=\frac{1.6389770508-1.2700958252}{0.4042877257+0.0001}
=0.9121969
$$
{% endraw %}

{% raw %}
$$
A_1=0.8607290
$$
{% endraw %}

{% raw %}
$$
A_2=\frac{0.6474609375-1.2700958252}{0.4042877257+0.0001}
=-1.5396979
$$
{% endraw %}

{% raw %}
$$
A_3=-0.2332281
$$
{% endraw %}

最终：

```text
advantages = [0.912197, 0.860729, -1.539698, -0.233228]
mean       = 0
```

含义：

- 正 advantage：该回答高于同题平均水平；
- 负 advantage：低于同题平均水平；
- 如果 4 条 reward 完全相同，所有 advantage 接近 0，这一组几乎没有相对学习信号。

## 4.7 从回答级 advantage 广播到 token

GRPO 的 `advantages.shape=[B×G]`，本例是 `[4]`。

代码：

```python
advantages.unsqueeze(1)
```

shape 变为 `[4,1]`，与 `[4,256]` 的 logp/ratio 运算时广播：

```text
回答 0 的 186 个有效 token 共用 A0= 0.912197
回答 1 的 184 个有效 token 共用 A1= 0.860729
回答 2 的 256 个有效 token 共用 A2=-1.539698
回答 3 的 193 个有效 token 共用 A3=-0.233228
```

所以 GRPO 没有 token 级 Critic advantage，但仍然计算逐 token policy loss。

## 5. GRPO 的 logp、ratio、KL 与 mask

## 5.1 三套 logp

| 变量 | shape | 来源 | 梯度 |
|---|---:|---|---|
| `old_per_token_logps` | `[4,256]` | rollout Actor | detach |
| `per_token_logps` | `[4,256]` | Current Actor 再 forward | 有 |
| `ref_per_token_logps` | `[4,256]` | Frozen Reference | 无 |

自回归对齐与 PPO 相同：completion token `P+t` 由 logits `P+t-1` 预测。

## 5.2 Reference、Old、Current、New 的时间身份
![GRPO 的 Reference / Old / Current / New 时间线](/images/minimind-grpo-cispo-real-complete/03-reference-policy-timeline.svg)


### 5.2.1 四个词分别指什么

设第 `k` 个训练 batch 开始时 Actor 参数为 $\theta^k$，冻结 Reference 参数为 $\theta_{ref}$。

| 名称 | 本项目中的实体 | 是否独立模型 | 是否更新 |
|---|---|---:|---|
| Reference | 从 DPO checkpoint 单独加载的 `ref_model` | 是，但冻结 | GRPO 全程不更新 |
| Old policy | rollout 返回的 `old_per_token_logps.detach()` | 否，只是 `[B×G,R]` 缓存 | 每次新 rollout 重新产生 |
| Current policy | 当前正在 forward/backward 的 `model` | 就是 Actor | optimizer step 时更新 |
| New policy | optimizer step 后的同一个 Actor | 否，不是新建模型 | step 后产生，并立即成为 Current |

初始化源码是：

```python
model, tokenizer = init_model(lm_config, base_weight, ...)
ref_model, _ = init_model(lm_config, base_weight, ...)
ref_model = ref_model.eval().requires_grad_(False)
optimizer = AdamW(model.parameters(), lr=3e-7)
```

历史 `base_weight=lifeos_agent_dpo_v1`，所以训练起点参数数值相同：

{% raw %}
$$
\theta^0=\theta_{ref}=\theta_{DPO}
$$
{% endraw %}

但二者是两个独立模型实例。只有 Actor 进入 optimizer；Reference 没有梯度、没有 optimizer、没有 `step()`。

Reference 参数始终不变：

{% raw %}
$$
\theta_{ref}^{start}=\theta_{ref}^{batch\ 1}=\cdots=\theta_{ref}^{end}
$$
{% endraw %}

不过每批生成的 token 不同，所以冻结 Reference 仍要对本批新 `outputs` forward，得到新的 `ref_per_token_logps`。这里“重新计算输出”不等于“更新参数”。

### 5.2.2 真实 GRPO/CISPO batch 的严格顺序

历史配置：

```text
B                  = 1
G                  = 4
accumulation_steps = 1
learning_rate      = 3e-7
loss_type          = cispo
rollout_engine     = torch
```

下面的**执行顺序**来自远程当前 `train_grpo.py`，四条回答、shape、logp、reward 和 advantage 来自真实 GRPO checkpoint 的只读 probe。probe 不执行 backward/optimizer step，所以文档不会虚构 step 后的新参数或新 logp。

#### 时刻 T0：batch 开始

```text
Actor 当前参数 = theta^k
Reference      = theta_ref（冻结 DPO 起点）
```

#### 时刻 T1：Actor 生成 4 条回答，当前参数被记录为 Old 数据

Torch rollout 用 $\theta^k$ 生成四条 completion：

```text
completion_ids       [4,256]
old_per_token_logps  [4,256]
```

数学上：

{% raw %}
$$
old\_logp_{i,t}^{k}=\log\pi_{\theta^k}(a_{i,t}\mid s_{i,t})
$$
{% endraw %}

这一步没有复制 `old_model`。Old 是采样时 logp 的冻结快照。

#### 时刻 T2：先评价四条完整回答

rollout 返回后，源码立即执行：

```text
rewards [4] = calculate_rewards(prompts, completions, reward_model)
```

得到真实 reward：

```text
[1.638977, 1.618164, 0.647461, 1.175781]
```

Reward Model 和规则奖励只评价文本，不更新 Actor 或 Reference。源码在 reward 完成后才进行带梯度的 Current Actor forward。

#### 时刻 T3：同一个 Actor 做带梯度的 Current forward

reward 计算完成后，源码用本地 Actor 对同一 `outputs` 再 forward：

```text
per_token_logps [4,256]
```

历史任务使用 Torch 引擎，而 Torch 引擎和训练端持有同一个 Actor 对象；从 T1 rollout 到 T3 Current forward 之间没有 optimizer step，所以：

{% raw %}
$$
\theta_{current}=\theta^k=\theta_{old\ rollout}
$$
{% endraw %}

真实探针生成索引 2 的最后 token“隐私”：

```text
old_logp     = -2.855593
current_logp = -2.855593
ratio        = exp(-2.855593 - (-2.855593)) = 1
```

#### 时刻 T4：冻结 Reference 对同一批 outputs forward

```text
ref_per_token_logps [4,256]
```

Reference 只提供 KL 锚点，不参与生成这四条回答，也不参与参数更新：

{% raw %}
$$
d_{i,t}=\log\pi_{ref}(a_{i,t}\mid s_{i,t})
-\log\pi_{\theta^k}(a_{i,t}\mid s_{i,t})
$$
{% endraw %}

真实“隐私”token 的非负 KL penalty 为 `0.090796`。这说明即使 `current=old`、ratio 恰好为 1，Current 与长期 Reference 仍可以不同。

#### 时刻 T5：组相对 advantage、mask 和 loss

T2 的四条 reward 转成组相对 advantage：

```text
[0.912197, 0.860729, -1.539698, -0.233228]
```

随后计算 ratio、Reference KL、CISPO token loss 和 scalar policy loss。到这个时刻，Actor 参数仍然是 $\theta^k$。

#### 时刻 T6：唯一一次 backward/optimizer step 产生 New policy

历史 `accumulation_steps=1`，因此本 batch 立即执行：

```text
loss.backward()
clip_grad_norm_(Actor, 1.0)
optimizer.step()     -> theta^k 变成 theta^{k+1}
scheduler.step()
optimizer.zero_grad()
```

刚 step 完的 $\theta^{k+1}$ 可以叫 New policy；它马上成为下一时刻的 Current policy。Reference 仍是 $\theta_{ref}$，本批的 `old_per_token_logps` 也不会被改写。

#### 时刻 T7：下一批重新定义 Old

下一批用 $\theta^{k+1}$ 生成新的四条回答：

{% raw %}
$$
old\_logp^{k+1}=\log\pi_{\theta^{k+1}}(a^{k+1}\mid s^{k+1})
$$
{% endraw %}

这时上一批的 New 成为下一批 rollout 的生成策略，而新 rollout 保存下来的 logp 又被称为 Old。完整循环是：

{% raw %}
$$
\theta^k
\xrightarrow{\text{rollout，保存 old logp}^k}
\theta^k
\xrightarrow{\text{backward + step}}
\theta^{k+1}
\xrightarrow{\text{下一批 rollout}}
old\_logp^{k+1}
$$
{% endraw %}

### 5.2.3 为什么本项目 GRPO 的 ratio 通常是 1

PPO 历史配置会对同一 rollout 做 `ppo_update_iters=2`：第一次 step 后，第二次还能用新 Current 对比固定 Old，所以 ratio 可能偏离 1。

当前 `train_grpo.py` 没有 PPO 那样的内层重复更新。每个 rollout 只在 optimizer step **之前**计算一次 Current logp；历史 Torch 引擎又与训练端共享 Actor 对象，因此本批观察到 `ratio=1` 是执行顺序的直接结果，不是近似值。

这并不表示 ratio 公式永远无用。以下情况会使它不等于 1：

1. rollout 由落后于训练端的外部策略生成；
2. 同一 rollout 被复用进行多次策略更新；
3. rollout 与 Current forward 之间发生了参数同步或其他更新。

### 5.2.4 Torch 与 SGLang 的更新时间差异

历史实跑为 `rollout_engine=torch`。`TorchRolloutEngine` 保存同一个模型对象的引用，optimizer 原地更新参数，因此下一批 rollout 自动读取最新 $\theta^{k+1}$。源码中每 `save_interval=250` 步显式调用一次 `update_policy(model)`，对 Torch 引擎只是重新设置同一对象引用，中间并不会真的停留在旧参数。

若使用 `rollout_engine=sglang`，rollout 在外部服务运行。`update_policy()` 会写出本地权重并调用 `/update_weights_from_disk`；按当前源码，显式同步发生在初始化、每个保存间隔和结束处。因此两个同步点之间：

```text
远程 rollout policy 可能是较旧参数 theta_rollout
本地 Current Actor   可能是较新参数 theta_current
old_logp = log pi_{theta_rollout}
current_logp = log pi_{theta_current}
ratio 可能不等于 1
```

这里 Old 必须以“真正生成 token 的策略”返回的 logp 为准，不能拿 Reference logp 代替。其正确性有一个必要前提：SGLang API 返回的 token logp 必须确实由远端本次采样所使用的那版权重和采样分布计算；否则 importance ratio 本身就失去可信基准。

### 5.2.5 一张最终时间表

| 时刻 | Actor/Current | Old logp | Reference | 会不会更新参数 |
|---|---|---|---|---|
| 初始化 | $\theta^0=\theta_{DPO}$ | 无 | $\theta_{ref}=\theta_{DPO}$ | 都只是加载 |
| T1 rollout 后 | 仍是 $\theta^k$ | 保存自 rollout Actor | 不变 | 否 |
| T2 reward 完成 | 仍是 $\theta^k$ | 固定 | 不变 | 否 |
| T3 Current forward | 仍是 $\theta^k$ | 固定 | 不变 | 否 |
| T4 Reference forward | 仍是 $\theta^k$ | 固定 | 参数不变；计算 ref logp | 否 |
| T5 loss 完成 | 仍是 $\theta^k$ | 固定 | 不变 | 否 |
| T6 optimizer step 后 | $\theta^{k+1}$（New→Current） | 仍是本批 $\theta^k$ 的缓存 | 不变 | 只更新 Actor |
| T7 下一批 rollout | $\theta^{k+1}$ | 重新保存自新 rollout | 不变 | rollout 本身不更新 |

## 5.3 Ratio

{% raw %}
$$
ratio_{i,t}=\exp(
logp_{current,i,t}-logp_{old,i,t}
)
$$
{% endraw %}

本次探针在更新前计算，因此所有有效位置 ratio 为 1。它是真实观察，但只代表更新前状态。

## 5.4 Reference KL

{% raw %}
$$
d_{i,t}=logp_{ref,i,t}-logp_{current,i,t}
$$
{% endraw %}

{% raw %}
$$
KL_{i,t}=e^{d_{i,t}}-d_{i,t}-1
$$
{% endraw %}

`beta=0.1` 控制 GRPO/CISPO 中 KL 惩罚强度。

## 5.5 Completion mask

GRPO 主循环：

1. 从 rollout 的 completion pad mask 开始；
2. 查找每条回答第一个 EOS；
3. 保留 `position <= eos_index`，所以 EOS 自身参与 loss；
4. EOS 后和 padding 位置为 0；
5. 没有 EOS 时保留到 tensor 末尾。

最终：

```text
completion_mask.shape = [4,256]
每行有效数           = [186,184,256,193]
```

## 6. 标准 GRPO 风格分支

项目 `loss_type="grpo"` 分支采用 PPO-clip 风格目标。

### 6.1 Clip ratio

{% raw %}
$$
ratio^{clip}_{i,t}=clip(ratio_{i,t},1-\epsilon,1+\epsilon)
$$
{% endraw %}

参数：

```text
epsilon = 0.2
区间    = [0.8,1.2]
```

### 6.2 Token loss

{% raw %}
$$
L^{GRPO}_{i,t}
=-\left[
\min(ratio_{i,t}A_i,ratio^{clip}_{i,t}A_i)
-\beta KL_{i,t}
\right]
$$
{% endraw %}

每个变量：

| 变量 | shape | 作用 |
|---|---:|---|
| `ratio` | `[4,256]` | current 相对 old 的概率倍数 |
| `ratio_clip` | `[4,256]` | 限制策略更新幅度 |
| `A_i` | `[4]→[4,1]` | 回答级方向和强度 |
| `KL` | `[4,256]` | 限制相对 Reference 漂移 |
| `token_loss` | `[4,256]` | mask 前逐 token loss |

### 6.3 真实 token 手算

生成索引 2 的最后 token 是“隐私”：

```text
A              = -1.539698
ratio          = 1
ratio_clip     = 1
KL             = 0.090796
beta           = 0.1
```

代入：

{% raw %}
$$
L^{GRPO}
=-[-1.539698-0.1\times0.090796]
=1.548778
$$
{% endraw %}

## 7. 本项目实际执行的 CISPO 分支

历史训练任务虽然叫 GRPO，但有效参数是：

```text
loss_type   = cispo
epsilon_high = 5.0
beta         = 0.1
```

因此必须区分：

- **GRPO 框架**：同题生成 G 条、组内 reward 标准化；
- **标准 GRPO 风格 loss**：ratio 双边 clip 后乘 advantage；
- **本项目实际 loss**：CISPO。

## 7.1 CISPO ratio

{% raw %}
$$
c_{i,t}=\min(ratio_{i,t},\epsilon_{high})
$$
{% endraw %}

代码还对 `c` 执行 `detach()`：ratio 只作为截断权重，主策略梯度直接通过 `logp_current`。

## 7.2 CISPO token loss

{% raw %}
$$
L^{CISPO}_{i,t}
=-\left[
c_{i,t}A_i\log\pi_\theta(a_{i,t}|s_{i,t})
-\beta KL_{i,t}
\right]
$$
{% endraw %}

变量：

| 变量 | 真实/参数 | 含义 |
|---|---|---|
| `c_i,t` | `min(ratio,5).detach()` | 截断重要性权重 |
| `A_i` | 组相对 advantage | 指定整条回答的学习方向 |
| `log πθ` | current token logp | 真正接收策略梯度 |
| `β` | 0.1 | KL 权重 |
| `KL_i,t` | 非负逐 token 估计 | 防止偏离 Reference |

## 7.3 真实 CISPO token 手算

仍使用生成索引 2 的最后 token：

```text
c             = 1
A             = -1.539698
current_logp  = -2.855593
KL            = 0.090796
beta          = 0.1
```

策略主项：

{% raw %}
$$
cA\log\pi
=1\times(-1.539698)\times(-2.855593)
\approx4.396751
$$
{% endraw %}

减 KL：

{% raw %}
$$
4.396751-0.1\times0.090796
=4.387671
$$
{% endraw %}

取外层负号：

{% raw %}
$$
L^{CISPO}=-4.387671
$$
{% endraw %}

Policy surrogate 可以为负。重点不是 loss 必须为正，而是梯度是否推动正 advantage token 概率上升、负 advantage token 概率下降。

## 8. 从 token loss 聚合为 scalar

先对每条回答做长度归一化：

{% raw %}
$$
L_i
=\frac{\sum_t m_{i,t}L_{i,t}}
{\sum_t m_{i,t}}
$$
{% endraw %}

再对 `B×G` 条回答取均值：

{% raw %}
$$
L_{policy}=\frac1{BG}\sum_iL_i
$$
{% endraw %}

先做每条回答内 mean，避免 256-token 回答天然比 184-token 回答权重大。

真实只读结果：

```text
标准 GRPO 风格 sequence losses:
[-0.905423, -0.854993, 1.546728, 0.237660]
policy_loss = 0.005993

CISPO sequence losses:
[0.797861, 0.769677, -1.414265, -0.174611]
policy_loss = -0.005334
```

训练时：

{% raw %}
$$
loss=\frac{policy\_loss+aux\_loss}{accumulation\_steps}
$$
{% endraw %}

本项目 `use_moe=0`、`accumulation_steps=1`，因此就是 policy loss。

## 9. GPU 上的完整执行过程
![CISPO 与 GPU 张量路径](/images/minimind-grpo-cispo-real-complete/04-cispo-gpu-loss.svg)


## 9.1 CPU 阶段

1. DataLoader 从 JSONL 取 conversations。
2. Python 执行 `conversations[:-1]`。
3. Chat template 生成 prompt 字符串。
4. Tokenizer 在 CPU 生成 `input_ids` 和 `attention_mask`。
5. `.to(cuda)` 把小型整数 Tensor 送入 GPU。

## 9.2 Rollout 阶段

1. `[1,561]` repeat-interleave 成 `[4,561]`。
2. Actor 在 `torch.no_grad()` 与 bfloat16 autocast 下自回归生成。
3. 每生成一个 token 都依赖前缀，因此 decoding 在时间维是逐步的。
4. GPU 保存/更新 KV cache，避免每一步重新计算全部前缀。
5. 最终得到 `[4,817]`，切出 completion `[4,256]`。
6. Actor 对完整 output 计算 old logp `[4,256]`。

## 9.3 Reward 阶段

1. Python 对 response 做长度、thinking、格式和重复规则评分。
2. Reward Model 驻留 GPU，以 FP16 对四个 answer 逐条打分。
3. 合并规则分和 RM score，得到 rewards `[4]`。

## 9.4 Target 构造阶段

这些运算都在 GPU 上，但开销远小于 Transformer：

```text
rewards [4]
→ grouped [1,4]
→ mean/std [1]
→ advantages [4]
→ broadcast [4,1]
```

## 9.5 Policy 与 Reference forward

1. Current Actor 对 `[4,817]` forward，保留 autograd graph。
2. 得到 logits `[4,817,6400]`，右移并 gather 成 current logps `[4,256]`。
3. Frozen Reference 在 `no_grad()` 中 forward，得到 ref logps `[4,256]`。
4. 计算 ratio、KL、token loss、mask 与 scalar loss。

## 9.6 Backward 与更新

真实训练而非 probe 时：

```python
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

`backward()` 从 scalar loss 沿 current logp、Transformer 激活回传到 Actor 参数。Reference 和 Reward Model 都冻结，不接收梯度。

## 9.7 Checkpoint

保存时：

1. 取 current policy state dict；
2. 转 FP16 并移动到 CPU；
3. 保存 `.pth`；
4. 保存 resume checkpoint 中的 optimizer/scheduler 状态；
5. rollout engine 在指定节点同步最新 policy。

## 10. 真实显存观察

PyTorch 探针记录的是 `memory_allocated()` 与 `max_memory_allocated()`：

### PPO 只读 forward

| 时点 | allocated / peak |
|---|---:|
| Actor + Reference + Critic | 808.4 MiB |
| rollout 后 peak | 963.4 MiB |
| Actor/Reference/Critic forward 后 peak | 981.3 MiB |
| 加载并运行 Reward Model 后 peak | 4,175.2 MiB |

### GRPO 只读 forward

| 时点 | allocated / peak |
|---|---:|
| Actor + Reference | 539.2 MiB |
| 4 路 rollout 后 peak | 773.5 MiB |
| Actor/Reference forward 后 peak | 870.7 MiB |
| 加载并运行 Reward Model 后 peak | 3,910.5 MiB |

注意：

- `allocated` 不等于 PyTorch `reserved`；
- 二者也不等于 `nvidia-smi` 显示的进程显存；
- probe 没有 backward、梯度、训练激活和 AdamW 状态；
- KV cache、训练 forward/backward、optimizer step 属于时间线上的不同阶段峰值，不能直接相加；
- Reward Model 参数通常常驻，RM no-grad 激活是临时量。

## 11. 历史训练事实

GRPO 历史训练：

```text
step             19,502 / 19,502
开始             2026-07-12 14:12:44
结束             2026-07-13 02:04:19
耗时             约 11 小时 51 分
最终权重         lifeos_agent_grpo_v1_768.pth
权重大小         约 132 MiB
```

日志第 1 步：

```text
Reward: -1.4673
KL_ref: 0.0005
Adv Std: 1.1545
Adv Mean: -0.0000
Actor Loss: 0.0097
Avg Response Len: 100.25
LR: 3e-7
```

最后一步：

```text
Reward: 2.0661
KL_ref: -0.0982
Adv Std: 1.1543
Adv Mean: 0.0000
Actor Loss: 0.0764
Avg Response Len: 202.75
LR: 3e-8
```

日志中的 `KL_ref` 是 `ref_logp-current_logp` 的采样均值，可以为负；loss 中的 `exp(d)-d-1` 才是非负 KL penalty。二者不是同一个指标。

训练完成、无 OOM/NaN 只证明工程流程完成，不证明模型回答质量合格。

## 12. PPO 与 GRPO/CISPO 总对照

| 对比项 | PPO | GRPO 框架 + CISPO loss |
|---|---|---|
| prompt 数据 | RLAIF prompt | 同一 RLAIF prompt |
| 每题生成 | 1 条 | 4 条 |
| Reward | 每条完整回答一个 | 每条完整回答一个 |
| Advantage | Critic + TD + GAE，token 级 | 组内 reward 标准化，回答级 |
| Critic | 有 | 无 |
| Returns | 有，Critic label | 无 |
| Value loss | 有 | 无 |
| Old logp | 有 | 有 |
| Reference KL | 有 | 有 |
| Policy loss | PPO clipped surrogate | CISPO；可切换 GRPO clip 分支 |
| 主要额外成本 | Critic forward/backward/optimizer | G 倍 rollout 与 RM 评分 |
| 适合前提 | Critic 能学到可信 value | 同题多回答能被 reward 可靠排序 |

## 13. 全变量账本

| 变量 | 来源 | Shape | 是否有梯度 | 下一步消费者 |
|---|---|---:|---|---|
| `prompts` | RLAIFDataset | `list[str]` | 否 | tokenizer |
| `input_ids` | tokenizer | `[B,P]` | 否 | rollout |
| `output_ids` | Actor generate | `[BG,P+R]` | 否 | logp/ref/reward |
| `completion_ids` | output slice | `[BG,R]` | 否 | decode/mask |
| `old_logps` | rollout Actor | `[BG,R]` | 否 | ratio |
| `current_logps` | Current Actor | `[BG,R]` | 是 | policy loss |
| `ref_logps` | Frozen Reference | `[BG,R]` | 否 | KL |
| `rewards` | rules + RM | `[BG]` | 否 | PPO token reward / GRPO group stats |
| `old_values` | PPO old Critic | `[B,R]` | 否 | TD/GAE/value clip |
| `token_rewards` | PPO terminal scatter | `[B,R]` | 否 | GAE |
| `raw_advantages` | PPO GAE | `[B,R]` | 否 | returns |
| `returns` | raw advantage + old value | `[B,R]` | target | Critic loss |
| `ppo_advantages` | standardized GAE | `[B,R]` | 权重 | Actor loss |
| `grouped_rewards` | GRPO view | `[B,G]` | 否 | mean/std |
| `grpo_advantages` | group normalize | `[BG]` | 权重 | token policy loss |
| `completion_mask` | EOS/padding logic | `[BG,R]` | 否 | loss reduction |
| `ratio` | exp(current-old) | `[BG,R]` | 依分支 | GRPO/CISPO |
| `per_token_kl` | current/ref | `[BG,R]` | 对 current 有 | policy loss |
| `per_token_loss` | policy objective | `[BG,R]` | 是 | masked reduction |
| `policy_loss` | sequence/group mean | `[]` | 是 | backward |
| `value_loss` | PPO only | `[]` | 是 | total loss |
| `total loss` | weighted sum | `[]` | 是 | backward |

## 14. 参数账本

| 参数 | PPO | GRPO 历史实验 | 作用 |
|---|---:|---:|---|
| `hidden_size` | 768 | 768 | Transformer 隐藏维度 |
| `num_hidden_layers` | 8 | 8 | Transformer 层数 |
| `max_seq_len` | 768 | 768 | prompt 最大宽度 |
| `max_gen_len` | 256（历史实跑） | 256 | 最大生成 token |
| `temperature` | 0.8 | 0.8 | rollout 采样随机性 |
| `learning_rate` | 3e-7 Actor | 3e-7 | policy 学习率 |
| `critic_learning_rate` | 5e-7 | 不适用 | Critic 学习率 |
| `gamma` | 1.0 | 不适用 | TD/GAE 折扣 |
| `lambda` | 0.95 | 不适用 | GAE 衰减 |
| `clip_epsilon` | 0.2 | GRPO 分支 0.2 | ratio 双边 clip |
| `cliprange_value` | 0.2 | 不适用 | Critic value clip |
| `vf_coef` | 0.5 | 不适用 | value loss 权重 |
| `kl_coef/beta` | 0.02 | 0.1 | Reference KL 权重 |
| `num_generations` | 1 | 4 | 每题 rollout 数 |
| `epsilon_high` | 不适用 | 5.0 | CISPO ratio 上界 |
| `grad_clip` | 1.0 | 1.0 | 梯度范数上限 |
| `dtype` | bfloat16 autocast | bfloat16 autocast | 混合精度 |
| `seed` | 42 | 42 | 随机可复现 |

## 15. 真实源码阅读顺序

1. `vendor/minimind-master/dataset/lm_dataset.py:195-224`：RLAIF、`[:-1]`、thinking 与 prompt。
2. `vendor/minimind-master/trainer/rollout_engine.py:23-36`：逐 token old logp。
3. `vendor/minimind-master/trainer/rollout_engine.py:39-92`：RolloutResult、repeat-interleave、generate 与 completion。
4. `vendor/minimind-master/trainer/train_ppo.py:84-151`：PPO rollout、mask、Critic、TD/GAE、returns。
5. `vendor/minimind-master/trainer/train_ppo.py:171-214`：PPO ratio、Actor loss、KL、value loss 与 backward。
6. `vendor/minimind-master/trainer/train_grpo.py:37-68`：GRPO reward。
7. `vendor/minimind-master/trainer/train_grpo.py:71-104`：GRPO rollout 与三套 logp。
8. `vendor/minimind-master/trainer/train_grpo.py:121-145`：组 advantage、mask、KL、GRPO/CISPO loss。
9. `vendor/minimind-master/trainer/train_grpo.py:205-243`：当前默认参数。
10. `real_ppo_probe.py`、`real_grpo_probe.py`：固定真实 checkpoint 的只读复算。

## 16. 最容易混淆的结论

1. Prompt 是完整历史，不只是最后一句“请回答这个问题”。
2. output 是 prompt+completion；completion 只是新生成部分。
3. old policy 不一定是一套驻留模型，本项目保存 old logp。
4. Reward Model 与 Critic 不同：前者评整条，后者逐状态估值。
5. token_rewards 是 GAE 输入；不是 GAE 输出。
6. normalized advantage 是 Actor 权重；returns 才是 Critic label。
7. 代码变量 `labels=output_ids[:,1:]` 是 gather token id，不是 Critic label。
8. GRPO 去掉 Critic，不等于去掉逐 token policy loss。
9. 历史任务叫 GRPO，但实际 `loss_type=cispo`。
10. ratio=1 是更新前状态，不代表算法没有作用。
11. policy surrogate 可以为负。
12. reward 高不等于事实正确；本真实样例已经证明这一点。
13. PyTorch allocated、reserved 与 nvidia-smi process memory 不相等。
14. 不同训练阶段的显存峰值不能简单相加。


## 17. 四条真实 GRPO 输出全文

### Generation 0：reward 1.638977，advantage 0.912197

```text
好的，用户问的是关于智能家居产品的安全性，以及如何确保客户的隐私和安全。首先，我需要确认用户的需求，他们可能关心产品的可靠性、性能和可靠性。

接下来，我要确保回答符合之前的对话，保持简洁和真诚。用户可能希望得到实际的建议，而不仅仅是理论上的回答。需要强调安全措施，比如加密、访问控制等。

还要注意用户可能没有提到的地方，比如是否在测试某些方面有问题，但根据之前的回答，用户更关注安全性和可靠性。因此，回答中应提到安全措施，比如加密、访问控制、访问控制等。
</think>

确保客户的隐私和安全是关键。需要明确安全措施的作用，比如加密和访问控制，同时提供可追溯的保护措施，如加密和访问控制。
```

### Generation 1：reward 1.618164，advantage 0.860729

```text
好的，用户问的是产品的安全性如何，需要解决数据加密技术、防火墙以及实时安全更新。首先，我需要确认用户的需求，他们可能是在测试我的反应，或者想了解具体的安全措施。

根据之前的对话，用户之前问过关于智能家居产品的问题，现在进一步探讨安全措施。我应该强调数据加密技术和防火墙，同时提供详细的安全说明。

用户可能希望了解具体的安全措施，比如数据加密、防火墙、实时安全更新等。需要确保回答既专业又易懂，同时保持简洁。
</think>

产品在使用过程中，需确保数据加密、防火墙和实时安全，同时采用多层次安全协议。例如，采用AES（高强度加密）和BES（安全系统防护）确保数据在传输和存储的同时保障数据。
```

### Generation 2：reward 0.647461，advantage -1.539698

```text
好的，用户问的是“如何处理智能家居产品中的安全问题，比如数据加密和实时安全更新”。首先，我需要确认用户的需求，他们可能关心数据安全和用户隐私，所以回答中需要强调数据安全和隐私保护。

接下来，我要确保回答符合系统规则，不透露身份信息，保持简洁。用户的问题涉及数据安全和隐私保护，所以需要强调用户保护措施，比如加密、访问控制、访问控制等。

然后，我需要确保回答符合系统规则，不透露身份信息，保持简洁。同时，要确保回答准确，不违反系统规则。可能需要提到一些常见的安全措施，比如加密、访问控制、访问控制等，但需要说明这些措施的必要性，并说明为什么这些措施是必要的。
</think>

产品在确保数据安全和隐私保护的前提下，可通过加密、访问控制、访问控制等技术实现。例如，可将敏感数据加密存储，同时限制外部访问控制，确保产品在生产中不被滥用。此外，用户可设置安全措施，如加密存储和身份验证，确保产品在运输中不被盗用。关键在于如何确保安全，同时保持对用户隐私
```

该回答达到 256 token 上限，没有生成 EOS。

### Generation 3：reward 1.175781，advantage -0.233228

```text
好的，用户问的是产品的安全性能如何。我需要先回顾之前的对话，确保回答连贯。之前讨论过智能家居产品的常见问题，比如数据加密和防火墙，现在用户想知道产品的安全性。

用户可能对安全性有疑问，或者想了解是否需要进一步的信息。需要确保回答准确，同时保持简洁。要提到具体的技术细节，比如加密和防火墙，这样用户能更全面地了解产品。

还要注意用户可能的深层需求，比如他们可能对系统安全性感兴趣，或者想了解如何应对类似问题。因此，在回答中可以加入一些实际的措施，比如定期进行安全演练，或者向用户推荐相关的防护措施。
</think>

这些产品的安全性能体现在安全性上，包括数据加密、防火墙和实时安全更新。若出现问题，应及时寻求专业人员的帮助。
```

## 18. 一句话总链路

```text
同一真实 prompt [1,561]
→ repeat_interleave(G=4) [4,561]
→ 4 条 completion [4,256]，有效长度 [186,184,256,193]
→ rewards [1.638977,1.618164,0.647461,1.175781]
→ mean 1.270096，std 0.404288
→ advantages [0.912197,0.860729,-1.539698,-0.233228]
→ 广播到 token [4,256]
→ current/old ratio + Reference KL
→ 实际 CISPO token loss [4,256]
→ 每条 masked mean → 4 条 mean
→ policy loss -0.005334（更新前只读 forward）
```

<!-- ORIGINAL_NOTE_END -->
