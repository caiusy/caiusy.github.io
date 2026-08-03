---
title: MiniMind PPO 完整拆解：真实输入、Token、Reward、Critic、GAE、Loss 与 GPU 全流程
date: 2026-08-01 23:19:00
updated: 2026-08-03 21:30:00
mathjax: true
description: >-
  基于远程 WSL 与真实 MiniMind checkpoint，从 991 字符 prompt、561 个 token、106 token rollout 出发，完整追踪 Reward、Critic、GAE、Reference、Old/Current/New policy、PPO 两次更新与 RTX 3090 Ti GPU 数据流。
categories:
  - AI与大模型
  - 深度学习
tags:
  - MiniMind
  - PPO
  - RLHF
  - Actor-Critic
  - GAE
  - Reference-Model
  - GPU
  - LLM-Alignment
type: deep-dive
difficulty: advanced
review_status: published
cover: /images/minimind-ppo-real-complete/01-ppo-full-pipeline.svg
---

这是一篇不省略中间步骤的 PPO 长文：从真实输入文字开始，保留原始 token、shape、Reward Model 输入输出、Critic value、TD/GAE、Reference、Old/Current/New policy、两次 optimizer update 与 GPU 路径的全部细节。正文完整收录原始笔记，只增加发布导航与配图。

<!-- more -->

<!-- ORIGINAL_NOTE_BEGIN: source content preserved; publication additions are image lines and MathJax wrappers -->
# MiniMind PPO：真实输入、Token、Reward、Critic、GAE、Loss 与 GPU 全流程

> 本文只讲 PPO。所有具体数值来自远程 WSL `wsl-dev`、MiniMind commit `4497610`、固定 `sample_index=0, seed=42` 的真实只读 forward。没有执行 backward、optimizer.step 或保存权重。

## 阅读目标

本文以一条真实输入为主线，逐步展示：原始文字 → chat template → token ID → Actor 输出 → Reward 输入输出 → Critic value → TD/GAE → Actor/Critic loss → scalar loss。每个变量都说明来源、shape、数值和梯度状态。

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
![PPO 真实样本端到端链路](/images/minimind-ppo-real-complete/01-ppo-full-pipeline.svg)


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

## 3.1 一次真实 PPO 样本从输入跑到输出

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

### 步骤 4：Actor 的实际输入和输出

Actor 输入：

```text
input_ids      [1,561]
attention_mask [1,561]
```

Actor 自回归生成 106 个 token：

```text
completion_ids [1,106]
output_ids     [1,667] = [1,561] prompt + [1,106] completion
```

实际生成全文：

```text
今天用户想知道如何在保证高效的同时，确保产品不被滥用，同时保护用户隐私和数据安全。需要平衡技术细节和用户隐私保护，确保用户在使用过程中不被滥用。
</think>

在保证高效的同时，需确保产品不被滥用，同时保护用户数据的安全与隐私，确保技术发展不被滥用。需要通过技术手段和用户隐私保护的平衡，确保产品不被滥用，同时保障用户数据的安全与隐私。
```

completion 的真实头尾 token：

| t | token id | token | old probability | old logp |
|---:|---:|---|---:|---:|
| 0 | 5640 | 今天 | 0.951236 | -0.049993 |
| 1 | 863 | 用户 | 0.039483 | -3.231891 |
| 2 | 1125 | 想 | 0.331798 | -1.103230 |
| 103 | 2457 | 隐私 | 0.625183 | -0.469710 |
| 104 | 302 | 。 | 0.872882 | -0.135955 |
| 105 | 2 | `<|im_end|>` | 0.411813 | -0.887186 |

### 步骤 5：Reward 的实际输入和输出

规则奖励输入是上面的完整 response，包括 thinking 和 answer。

Reward Model 的输入不是整段 response 原样，而是：

```text
messages = 前面 5 条 prompt messages

answer =
在保证高效的同时，需确保产品不被滥用，同时保护用户数据的安全与隐私，
确保技术发展不被滥用。需要通过技术手段和用户隐私保护的平衡，
确保产品不被滥用，同时保障用户数据的安全与隐私。
```

也就是先按 `</think>` 切分，只把其后的 answer 交给 RM。

`LMForRewardModel.get_score()` 随后把前 4 条历史消息压成 `history_text`，把最后一个 user 问题写成“我的新问题”，再构造两条 RM 对话：

```text
RM eval_messages[0].role = user
RM eval_messages[0].content =
  user: <最初角色信息>
  assistant: <完整张明/刘琳历史对话>
  user: 基于以上对话提出一个问题。
  assistant: 这些智能家居产品需要哪些前提条件才能够使用？
  以上是对话历史。我的新问题是：
  请回答这个问题。

RM eval_messages[1].role = assistant
RM eval_messages[1].content = <上面切出的 answer>
```

InternLM2 Reward Model 使用自己的 tokenizer，不是 MiniMind 的 6400 词表。真实编码结果：

```text
rm_input_ids.shape      = [1,523]
rm_attention_mask.shape = [1,523]
first 12 ids = [1,92543,1008,364,1008,334,262,70513,68544,68968,68347,68360]
last  12 ids = [60353,68405,69572,68508,68398,73820,60510,70713,60355,92542,364,92527]
reward_token_id = 92527
```

最后的 `92527` 是 `<|reward|>`。Reward Model 对 `[1,523]` forward，并取这个末尾 reward token 对应的 scalar score：`-0.55029296875`。

Reward 各项输出：

| 项 | 输入检查 | 输出 |
|---|---|---:|
| 长度 | 完整 response 在 20–800 字符 | +0.5 |
| thinking | `</think>` 前内容在 20–300 字符 | +1.0 |
| 格式 | 恰好一个 `</think>` | +0.25 |
| 重复惩罚 | answer 的 3-gram | -0.0 |
| Reward Model | `messages + answer` | -0.55029296875 |
| final reward | 上述相加 | 1.19970703125 |

### 步骤 6：Critic 的实际输入和 value 输出

Critic 的输入不是只有 answer，而是完整 Actor output：

```text
critic input_ids      = output_ids [1,667]
critic attention_mask = full_mask [1,667]
```

网络内：

```text
[1,667]
→ embedding/8层 Transformer
→ hidden [1,667,768]
→ value_head Linear(768,1)
→ values_seq [1,667]
→ gather logp_pos=560...665
→ old_resp_values [1,106]
```

选取的真实 value：

| t | 状态含义 | old value |
|---:|---|---:|
| 0 | 看完 prompt、准备生成“今天” | 0.654884 |
| 1 | 已生成“今天”、准备生成“用户” | 0.711958 |
| 2 | 已生成“今天用户”、准备生成“想” | 0.597131 |
| 103 | 准备生成“隐私” | 0.729777 |
| 104 | 准备生成“。” | 0.646647 |
| 105 | 准备生成 `<|im_end|>` | 0.716022 |

Critic 的 value 不是概率，也不是 Reward Model score。它预测“从当前生成状态继续下去，预计能得到多少未来回报”。

### 步骤 7：reward 怎样进入 GAE

```text
final reward   [1]     = [1.199707]
token_rewards  [1,106] = [0,0,...,0,1.199707]
old_values     [1,106]
```

最后三个位置逐个计算：

```text
t=105:
delta = 1.199707 + 1×0 - 0.716022 = 0.483685
A105  = 0.483685
return= 0.483685 + 0.716022       = 1.199707

t=104:
delta = 0 + 1×0.716022 - 0.646647 = 0.069375
A104  = 0.069375 + 0.95×0.483685  = 0.528876
return= 0.528876 + 0.646647        = 1.175523

t=103:
delta = 0 + 1×0.646647 - 0.729777 = -0.083130
A103  = -0.083130 + 0.95×0.528876 = 0.419302
return= 0.419302 + 0.729777        = 1.149079
```

### 步骤 8：真实 PPO token 账本

| t | token | old logp | ref logp | ratio | KL | old value | reward | raw A | return | norm A | actor token loss | value token loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 今天 | -0.049993 | -9.877064 | 1 | 8.827125 | 0.654884 | 0 | 0.024116 | 0.678999 | -0.586995 | 0.586995 | 0.000291 |
| 1 | 用户 | -3.231891 | -4.410814 | 1 | 0.486533 | 0.711958 | 0 | -0.034693 | 0.677265 | -1.051236 | 1.051236 | 0.000602 |
| 2 | 想 | -1.103230 | -1.711979 | 1 | 0.152780 | 0.597131 | 0 | 0.084351 | 0.681482 | -0.111488 | 0.111488 | 0.003558 |
| 103 | 隐私 | -0.469710 | -0.360522 | 1 | 0.006184 | 0.729777 | 0 | 0.419302 | 1.149079 | 2.532647 | -2.532647 | 0.087907 |
| 104 | 。 | -0.135955 | -0.090445 | 1 | 0.001052 | 0.646647 | 0 | 0.528876 | 1.175523 | 3.397639 | -3.397639 | 0.139855 |
| 105 | EOS | -0.887186 | -0.018923 | 1 | 0.514506 | 0.716022 | 1.199707 | 0.483685 | 1.199707 | 3.040898 | -3.040898 | 0.116976 |

### 步骤 9：真实 scalar 输出

106 个 token 经 mask 求均值：

```text
raw advantage mean = 0.098474
raw advantage std  = 0.126677
ratio mean         = 1.0
actor loss         ≈ 0.000000036
Reference KL mean  = 0.335082
policy loss        = 0.006702
value loss         = 0.012872
total loss         = 0.013138
```

这就是这条真实输入在“更新前 PPO forward”中的完整输出链。

## 4. JSONL 到 prompt tensor

### 4.1 为什么使用 `conversations[:-1]`

真实数据最后一条是：

```json
{"role": "assistant", "content": ""}
```

`RLAIFDataset.create_chat_prompt()` 使用：

```python
tokenizer.apply_chat_template(
    conversations[:-1],
    tokenize=False,
    open_thinking=use_thinking,
    add_generation_prompt=True,
)
```

`[:-1]` 删除空 assistant 占位。PPO/GRPO 要让当前 Actor 在线生成回答；如果数据里已经提供标准 assistant 文本并对它做逐 token 模仿，那更接近 SFT。

### 4.2 真实 prompt

固定：

```text
sample_index   = 0
seed           = 42
thinking_ratio = 0.9
```

得到：

```text
prompt 类型             str
prompt 字符数           991
input_ids.shape         [1,561]
attention_mask.shape    [1,561]
```

此时还没有 completion、reward、advantage 或 loss。

## 5. PPO 完整回顾

## 5.1 PPO 中的五个角色

| 对象 | 本项目实现 | 输出 | 是否训练 | 作用 |
|---|---|---|---|---|
| Current Actor | `MiniMindForCausalLM` | token logits / current logp | 是 | 生成并学习回答 |
| Old policy | rollout 时保存的 old logp | `[B,R]` | 否 | 表示采样数据时的策略 |
| Reference | 冻结 DPO checkpoint | ref logp `[B,R]` | 否 | KL 锚点，限制长期漂移 |
| Reward Model | InternLM2 1.8B RM | 每条回答一个 score | 否 | 评价完整回答 |
| Critic | MiniMind 主干 + value head | value `[B,R]` | 是 | 预测每个状态未来回报 |

Old policy 与 Reference 不是同一个东西：

- old policy 比较“一次 rollout 采样时”和“当前更新时”的变化；
- Reference 比较 Actor 与 PPO 开始前稳定模型的长期变化。

### 5.1.1 先把四个最容易混淆的词钉死
![Reference / Old / Current / New 时间线](/images/minimind-ppo-real-complete/02-reference-policy-timeline.svg)


设第 `k` 个训练 batch 开始时，Actor 参数为 $\theta^{k}$，Critic 参数为 $\phi^{k}$，冻结 Reference 参数为 $\theta_{ref}$。

| 名称 | 在本项目里到底是什么 | 是否单独占一套可训练参数 | 什么时候改变 |
|---|---|---:|---|
| Reference | 从 `lifeos_agent_dpo_v1_768.pth` 单独加载的完整模型，参数记为 $\theta_{ref}$ | 是一套独立模型，但冻结 | PPO 全程不改变 |
| Old policy | 本轮 rollout 时 Actor 对已生成 token 给出的 `old_resp_logp`，即 $\log\pi_{\theta^k}(a_t\mid s_t)$ 的缓存张量 | **不是**第四套模型 | 每次新 rollout 都重新生成并覆盖语义上的“old”基准 |
| Current policy | 正在参加当前 forward/backward 的 Actor；参数就是此刻内存中的 $\theta$ | 就是 Actor 本身 | 每次 `actor_optimizer.step()` 后改变 |
| New policy | optimizer step 完成后的 Current Actor，例如 $\theta^{k,1}$、$\theta^{k,2}$ | **不是**另一套模型 | `actor_optimizer.step()` 执行瞬间产生；随后它就成为新的 Current |

因此，“旧、当前、新”是同一 Actor 在不同时间点的身份，不代表 GPU 上同时常驻三套 Actor：

{% raw %}
$$
\underbrace{\theta^k}_{\text{rollout 时的 Actor}}
\xrightarrow{\text{保存 old logp}}
\underbrace{\theta^k}_{\text{第 1 次更新前的 Current}}
\xrightarrow{optimizer.step}
\underbrace{\theta^{k,1}}_{\text{New，立刻成为 Current}}
\xrightarrow{optimizer.step}
\underbrace{\theta^{k,2}}_{\text{再次更新后的 New}}
$$
{% endraw %}

Reference 不在这条更新链上：

{% raw %}
$$
\theta_{ref}^{start}=\theta_{ref}^{step\ 1}=\cdots=\theta_{ref}^{end}
$$
{% endraw %}

### 5.1.2 Reference 从哪里来，为什么最初看起来和 Actor 一样

PPO 初始化时源码连续执行两次 `init_model(lm_config, base_weight)`：

```python
actor_model, tokenizer = init_model(lm_config, base_weight, ...)
ref_model, _ = init_model(lm_config, base_weight, ...)
ref_model = ref_model.eval().requires_grad_(False)
```

历史任务中 `base_weight=lifeos_agent_dpo_v1`，所以训练刚开始时：

{% raw %}
$$
\theta^{0}=\theta_{ref}=\theta_{DPO}
$$
{% endraw %}

这里的等号表示**初始参数数值相同**，并不表示二者是同一个 Python 模型或共享同一块参数。随后只有 Actor 被交给：

```python
actor_optimizer = AdamW(actor_model.parameters(), lr=3e-7)
```

Reference 没有 optimizer、`requires_grad=False`，也在 `no_grad()` 中 forward，所以：

- Actor 每次 optimizer step 都可能变化；
- Reference 永远保留 PPO 开始前的 DPO 行为；
- PPO 越往后训练，Current Actor 与 Reference 通常越不相同；
- Reference 的**参数**固定，但每个 batch 的 prompt/completion 不同，所以每轮仍要对新 `output_ids` 计算新的 `ref_resp_logp` 张量。

Reference 的作用是长期锚点。Old policy 约束“不要相对刚才采样时跳太远”，Reference KL 约束“不要相对 PPO 起点长期漂移太远”。

### 5.1.3 一个真实 PPO batch 的严格执行顺序

历史配置为：

```text
B                  = 1
mini_batch_size    = 1
ppo_update_iters   = 2
accumulation_steps = 1
Actor learning rate  = 3e-7
Critic learning rate = 5e-7
rollout_engine       = torch
```

下面的**执行顺序**来自远程当前 `train_ppo.py`，形状和 token 数值来自真实 PPO checkpoint 的只读 probe。probe 没有执行 backward/optimizer step，因此它能真实证明“更新前 current=old”，但不能提供第一次 step 后的新 logp；后者只写公式，不造数。

第 `k` 个 batch 按下面顺序执行。

#### 时刻 T0：batch 开始

```text
Actor 当前参数     = theta^k
Critic 当前参数    = phi^k
Reference 参数     = theta_ref（冻结，仍是 DPO 起点）
```

#### 时刻 T1：Current Actor 先生成 rollout，并把这一刻冻结成 Old 数据

`TorchRolloutEngine` 持有同一个 Actor 对象，用 $\theta^k$ 生成真实的 106 个 completion token，并保存：

```text
completion_ids [1,106]
old_resp_logp  [1,106] = log pi_{theta^k}(a_t | s_t)
```

此时没有复制出 `old_model`。所谓 Old policy，是 `old_resp_logp.detach()` 所代表的采样时刻策略。

#### 时刻 T2：先算整条回答 reward，再固定本轮 Critic/Reference 目标

rollout 返回后，源码先执行：

```text
rewards [1] = calculate_rewards(prompts, responses_text, reward_model)
```

本真实回答得到 `reward=1.199707`。随后建立 completion mask，再进入 `torch.no_grad()`，严格按源码顺序得到：

```text
old_resp_values [1,106] = V_{phi^k}(s_t)
ref_resp_logp   [1,106] = log pi_ref(a_t | s_t)
token_rewards   [1,106]
advantages      [1,106]
returns         [1,106]
```

这些张量在本 batch 的两次 PPO update 中保持不变。特别注意：

- `old_resp_values` 是 Critic 在更新前的缓存，不会随第一次 Critic step 改写；
- `ref_resp_logp` 是冻结 Reference 对本轮固定 `output_ids` 的计算结果；
- advantages 和 returns 只计算一次，两次 update 共用。

#### 时刻 T3：第 1 次 PPO update，更新前 Current 与 Old 相同

每个 minibatch 中，源码先让 Current Critic forward 得到 `mb_resp_values`，再让 Current Actor 对同一批 `output_ids` forward 得到 `mb_resp_logp`。此刻尚未执行任何 Actor/Critic step，所以：

{% raw %}
$$
\theta_{current}=\theta^k=\theta_{old\ rollout}
$$
{% endraw %}

真实 EOS token：

```text
old_logp     = -0.887186
current_logp = -0.887186
ratio        = exp(-0.887186 - (-0.887186)) = 1
ref_logp     = -0.018923
Reference KL = 0.514506
```

随后按顺序执行：

```text
loss.backward()
clip_grad_norm_(Actor, 1.0)
clip_grad_norm_(Critic, 1.0)
actor_optimizer.step()   -> theta^k 变成 theta^{k,1}
critic_optimizer.step()  -> phi^k   变成 phi^{k,1}
actor_scheduler.step()
critic_scheduler.step()
actor_optimizer.zero_grad()
critic_optimizer.zero_grad()
```

`theta^{k,1}` 在刚产生时可以叫 New policy；从下一行代码开始，它就是 Current policy。

#### 时刻 T4：第 2 次 PPO update，Current 已经是新的，Old 仍是 T1 的缓存

源码再次按“Current Critic forward → Current Actor forward”的顺序重算当前值。Actor ratio 使用：

{% raw %}
$$
ratio_t^{(2)}
=\exp\left(
\log\pi_{\theta^{k,1}}(a_t\mid s_t)
-\underbrace{\log\pi_{\theta^k}(a_t\mid s_t)}_{old\_resp\_logp\text{，仍固定}}
\right)
$$
{% endraw %}

以真实 EOS 的 old logp 写开就是：

{% raw %}
$$
ratio_{EOS}^{(2)}
=\exp\left(\log\pi_{\theta^{k,1}}(EOS\mid s)-(-0.887186)\right)
$$
{% endraw %}

这一次 ratio 才可能不等于 1。只读探针没有执行 `optimizer.step()`，因此没有真实的 $\log\pi_{\theta^{k,1}}$ 数值；这里保留公式，不伪造更新后结果。

第二次 backward/step 后：

```text
Actor  = theta^{k,2}
Critic = phi^{k,2}
```

若 `approx_kl > early_stop_kl`，源码会把该次 loss 乘 0，完成 DDP forward-backward 闭环但不做有效参数更新。这是提前停止的保护分支。

#### 时刻 T5：进入下一个 batch，上一轮 New 变成下一轮 Old 的来源

令：

{% raw %}
$$
\theta^{k+1}=\theta^{k,2},\qquad \phi^{k+1}=\phi^{k,2}
$$
{% endraw %}

下一批先用 $\theta^{k+1}$ 重新生成文本和 logp。新缓存的：

{% raw %}
$$
old\_resp\_logp^{k+1}=\log\pi_{\theta^{k+1}}(a_t^{k+1}\mid s_t^{k+1})
$$
{% endraw %}

才成为下一批的 Old 基准。上一批的 `old_resp_logp^k` 不会被拿来和下一批不同的 token 比较。

### 5.1.4 用一张表看清“谁在什么时候更新”

| 时刻 | Actor/Current | Old logp | Critic/Current value | Old value | Reference |
|---|---|---|---|---|---|
| 初始化 | $\theta^0=\theta_{DPO}$ | 尚无 | $\phi^0$ | 尚无 | $\theta_{ref}=\theta_{DPO}$，冻结 |
| T1 rollout 后 | 仍是 $\theta^k$ | 保存自 $\theta^k$ | 仍是 $\phi^k$ | 尚无 | 参数不变 |
| T2 目标计算后 | 仍是 $\theta^k$ | 固定 | 仍是 $\phi^k$ | 保存自 $\phi^k$ | 参数不变；得到本批 ref logp |
| 第 1 次 step 后 | $\theta^{k,1}$ | 仍是 $\theta^k$ 的缓存 | $\phi^{k,1}$ | 仍是 $\phi^k$ 的缓存 | 不变 |
| 第 2 次 step 后 | $\theta^{k,2}$ | 仍是 $\theta^k$ 的缓存 | $\phi^{k,2}$ | 仍是 $\phi^k$ 的缓存 | 不变 |
| 下一批 rollout 后 | $\theta^{k+1}$ | 重新保存自 $\theta^{k+1}$ | $\phi^{k+1}$ | 随后重新保存 | 不变 |

### 5.1.5 为什么必须同时有 Old 和 Reference

它们回答的是两个不同问题：

{% raw %}
$$
ratio_t=\frac{\pi_{current}(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
$$
{% endraw %}

回答：“这次参数更新相对产生本批数据的策略走了多远？”它服务于 PPO clip 和近端更新。

{% raw %}
$$
KL_t=f\!\left(\log\pi_{ref}(a_t\mid s_t)-\log\pi_{current}(a_t\mid s_t)\right)
$$
{% endraw %}

回答：“当前 Actor 相对 PPO 开始前的 DPO 模型累计漂移了多远？”它服务于长期行为约束。

即使第 1 次 update 的 `ratio=1`，Reference KL 也不一定为 0。本真实 EOS 上就是：

```text
current = old  -> ratio = 1
current != ref -> Reference KL = 0.514506
```

原因是探针加载的是已经训练完成的 PPO Actor，而 Reference 仍是 DPO checkpoint；探针中的 Current/Old 在当前 rollout 时相同，但 Current 与长期 Reference 已经不同。

### 5.1.6 `update_policy()` 到底什么时候更新 Rollout Actor

历史训练用 `rollout_engine=torch`。`TorchRolloutEngine` 保存的是 Actor 模型对象的引用：

```python
self.policy_model = actor_model
```

`actor_optimizer.step()` 会原地修改这个对象的参数，所以每次 step 后，下一次 Torch rollout 自动看到最新 Actor。源码虽然只在训练开始、每 `save_interval=250` 步以及训练结束显式调用 `rollout_engine.update_policy(actor_model)`，但对 Torch 引擎而言，这个调用只是重新赋同一个对象引用，不代表中间 249 步使用旧权重。

如果改用 `rollout_engine=sglang`，情况不同：SGLang 是外部服务，`update_policy()` 会先把权重写到共享目录，再请求 `/update_weights_from_disk`。按当前源码，它只在初始化、保存间隔和结束处显式同步，因此两个同步点之间外部 rollout 服务可能落后于训练 Actor。在远端 API 返回的 token logp 确实由“实际执行本次采样的那版权重和分布”计算这一必要前提下，`old_logp` 才正确表示生成这批 token 的远程策略；Current forward 使用本地最新 Actor，第一次比较时 ratio 也可能不再等于 1。

## 5.2 PPO 真实 rollout

Actor 使用 `temperature=0.8` 采样，得到：

```text
prompt_ids       [1,561]
output_ids       [1,667]
completion_ids   [1,106]
old_logps        [1,106]
completion_mask  [1,106]
valid tokens     106
```

关系是沿 sequence 维拼接：

{% raw %}
$$
output\_ids=concat(prompt\_ids,completion\_ids)
$$
{% endraw %}

不是 Tensor 数值相加。

真实生成文本为：

```text
今天用户想知道如何在保证高效的同时，确保产品不被滥用，同时保护用户隐私和数据安全。
需要平衡技术细节和用户隐私保护，确保用户在使用过程中不被滥用。
</think>

在保证高效的同时，需确保产品不被滥用，同时保护用户数据的安全与隐私……
```

这条回答没有直接回答稳定 Wi-Fi 和安装说明，因此语义偏离。

## 5.3 自回归位置对齐

第 `t` 个 completion token 位于完整 output 的 `P+t`，由 logits 的 `P+t-1` 预测：

```text
completion token 0：output[561] ← logits[560] 预测
completion token 1：output[562] ← logits[561] 预测
...
completion token 105：output[666] ← logits[665] 预测
```

代码等价于：

```python
labels = output_ids[:, 1:]
logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(R)
selected_logps = all_shifted_logps.gather(1, logp_pos)
```

因此：

```text
logits       [B,L,V]
右移后 logits [B,L-1,V]
实际 token id [B,L-1]
gather 后     [B,L-1]
再选 response [B,R]
```

## 5.4 probability、logp 与 old logp

语言模型计算：

{% raw %}
$$
p_t=P(a_t\mid s_t)
$$
{% endraw %}

其中：

- `s_t`：prompt 加已经生成的 token；
- `a_t`：当前实际生成 token；
- `p_t`：Actor 给这个实际 token 的条件概率。

对数概率：

{% raw %}
$$
logp_t=\log p_t
$$
{% endraw %}

真实 PPO token：

| t | token | old logp | `exp(old_logp)` |
|---:|---|---:|---:|
| 0 | 今天 | -0.049993 | 0.951236 |
| 1 | 用户 | -3.231891 | 0.039483 |
| 2 | 想 | -1.103230 | 0.331798 |
| 105 | `<|im_end|>` | -0.887186 | 0.411813 |

实现注意：`generate()` 用温度 0.8 采样，但 `compute_per_token_logps()` 对原始 logits 做 `log_softmax`，没有再次除以 0.8。因此保存的 old logp 对应基础 logits，并不严格等于温度化采样分布的 logp。

## 5.5 PPO reward

本项目终局奖励：

{% raw %}
$$
R_{final}=R_{length}+R_{think}+R_{format}-R_{repeat}+R_{RM}
$$
{% endraw %}

每个变量：

| 变量 | 真实值 | 来源 |
|---|---:|---|
| `R_length` | `+0.5` | response 字符数位于 20–800 |
| `R_think` | `+1.0` | thinking 内容长度位于 20–300 |
| `R_format` | `+0.25` | 恰好一个 `</think>` |
| `R_repeat` | `0.0` | answer 的 3-gram 重复惩罚 |
| `R_RM` | `-0.55029296875` | InternLM2 Reward Model |

代入：

{% raw %}
$$
R_{final}=0.5+1.0+0.25-0-0.55029296875
=1.19970703125
$$
{% endraw %}

规则评分完整 response；若存在 `</think>`，Reward Model 只评分 `</think>` 后的 answer 加 prompt messages。

## 5.6 从 sequence reward 到 token rewards

PPO 创建：

```text
token_rewards.shape = [1,106]
token_rewards       = [0,0,...,0,1.199707]
```

只有最后一个有效 token 得到终局 reward：

{% raw %}
$$
r_{105}=1.19970703125,\qquad r_t=0\;(t\lt 105)
$$
{% endraw %}

`token_rewards` 是 GAE 的输入，不是 GAE 输出。

## 5.7 Critic value

Critic 对每个生成前状态预测未来总回报：

{% raw %}
$$
V_{old,t}=V_{\phi_{old}}(s_t)
$$
{% endraw %}

完整序列一次 forward：

```text
input_ids     [1,667]
hidden        [1,667,768]
value_head    Linear(768,1)
values        [1,667]
response value gather 后 [1,106]
```

它不是逐 token 运行 106 次 Transformer。

## 5.8 TD residual
![Critic、TD、GAE 与 Returns 张量流](/images/minimind-ppo-real-complete/03-critic-gae-tensors.svg)


公式：

{% raw %}
$$
\delta_t=r_t+\gamma V_{old,t+1}-V_{old,t}
$$
{% endraw %}

变量：

| 变量 | 含义 | 本项目参数/来源 |
|---|---|---|
| `r_t` | 当前 token 后立即得到的 reward | `token_rewards` |
| `V_old,t` | Critic 对当前状态的旧预测 | old Critic forward |
| `V_old,t+1` | 下一状态的旧预测 | 同一 value tensor 下一位置 |
| `gamma` | 未来回报折扣 | `1.0` |
| `delta_t` | 当前 value 的一步预测误差 | 由上述量计算 |

终止位置没有下一状态，令 `V_{106}=0`。

## 5.9 GAE

递推公式：

{% raw %}
$$
A_t^{raw}=\delta_t+\gamma\lambda A_{t+1}^{raw}
$$
{% endraw %}

项目参数：

```text
gamma  = 1.0
lambda = 0.95
```

`lambda` 决定后续 TD residual 向前传播的强度。GAE 从后向前做 Tensor 递推，不是神经网络的 `loss.backward()`。

### 最后三个真实 token

真实值：

| t | token | `V_old,t` | `r_t` |
|---:|---|---:|---:|
| 103 | 隐私 | 0.729777 | 0 |
| 104 | 。 | 0.646647 | 0 |
| 105 | `<|im_end|>` | 0.716022 | 1.199707 |

终止位置：

{% raw %}
$$
\delta_{105}=1.199707+0-0.716022=0.483685
$$
{% endraw %}

{% raw %}
$$
A_{105}^{raw}=0.483685
$$
{% endraw %}

倒数第二个位置：

{% raw %}
$$
\delta_{104}=0+0.716022-0.646647=0.069375
$$
{% endraw %}

{% raw %}
$$
A_{104}^{raw}=0.069375+0.95\times0.483685=0.528876
$$
{% endraw %}

再前一个位置：

{% raw %}
$$
\delta_{103}=0+0.646647-0.729777=-0.083130
$$
{% endraw %}

{% raw %}
$$
A_{103}^{raw}=-0.083130+0.95\times0.528876=0.419302
$$
{% endraw %}

## 5.10 Returns 与 advantage 标准化

Critic 的目标：

{% raw %}
$$
Return_t=A_t^{raw}+V_{old,t}
$$
{% endraw %}

真实值：

```text
Return103 = 0.419302 + 0.729777 = 1.149079
Return104 = 0.528876 + 0.646647 = 1.175523
Return105 = 0.483685 + 0.716022 = 1.199707
```

106 个位置的 raw advantage 统计：

```text
mean = 0.098474
std  = 0.126677
```

Actor 使用标准化 advantage：

{% raw %}
$$
\hat A_t=\frac{A_t^{raw}-\mu_A}{\sqrt{Var(A^{raw})+10^{-8}}}
$$
{% endraw %}

EOS：

{% raw %}
$$
\hat A_{105}
=\frac{0.483685-0.098474}{0.126677}
=3.040898
$$
{% endraw %}

必须区分：

- raw advantage：构造 returns；
- normalized advantage：Actor loss 权重；
- returns：Critic label；
- current values：Critic prediction。

## 5.11 PPO Actor ratio 与 clip

逐 token ratio：

{% raw %}
$$
ratio_t
=\frac{\pi_{current}(a_t|s_t)}{\pi_{old}(a_t|s_t)}
=\exp(logp_{current,t}-logp_{old,t})
$$
{% endraw %}

参数：

```text
clip_epsilon = 0.2
允许区间       [0.8,1.2]
```

Actor token loss：

{% raw %}
$$
L_{actor,t}=\max\left(
-\hat A_t ratio_t,
-\hat A_t clip(ratio_t,0.8,1.2)
\right)
$$
{% endraw %}

真实探针在 optimizer step 前重新计算 current logp，因此 current=old、ratio=1。例如 EOS：

```text
old_logp     = -0.887186
current_logp = -0.887186
ratio        = exp(0) = 1
```

执行 optimizer step 后再用同一 rollout 更新时，current 才可能与 old 分开。本文不虚构更新后的逐 token ratio。

## 5.12 Reference KL

令：

{% raw %}
$$
d_t=logp_{ref,t}-logp_{current,t}
$$
{% endraw %}

本项目使用非负估计：

{% raw %}
$$
KL_t=e^{d_t}-d_t-1
$$
{% endraw %}

EOS 真实值：

```text
ref_logp     = -0.018923
current_logp = -0.887186
d            = 0.868264
KL           = exp(0.868264)-0.868264-1
             = 0.514506
```

PPO 的 `kl_coef=0.02`。

## 5.13 Critic clipped value loss

项目参数：

```text
cliprange_value = 0.2
vf_coef         = 0.5
```

公式：

{% raw %}
$$
E_{1,t}=(V_{current,t}-Return_t)^2
$$
{% endraw %}

{% raw %}
$$
V_{clip,t}=clip(V_{current,t},V_{old,t}-0.2,V_{old,t}+0.2)
$$
{% endraw %}

{% raw %}
$$
E_{2,t}=(V_{clip,t}-Return_t)^2
$$
{% endraw %}

{% raw %}
$$
L_{value,t}=\frac12\max(E_{1,t},E_{2,t})
$$
{% endraw %}

探针更新前 `V_current=V_old`。EOS：

{% raw %}
$$
L_{value,105}=\frac12(0.716022-1.199707)^2=0.116976
$$
{% endraw %}

## 5.14 PPO total loss

{% raw %}
$$
L_{policy}=L_{actor}+0.02L_{reference\ KL}
$$
{% endraw %}

{% raw %}
$$
L_{total}=L_{policy}+0.5L_{value}+L_{MoE\ aux}
$$
{% endraw %}

本次 `use_moe=0`，所以 aux loss 为 0。真实只读 scalar：

```text
actor_loss  ≈ 3.60e-8
KL mean      = 0.335082
policy_loss  = 0.006702
value_loss   = 0.012872
total_loss   = 0.006702 + 0.5×0.012872
             = 0.013138
```

Actor loss 接近 0 的原因是 ratio=1 且 normalized advantage 的 masked mean 约为 0；不是 Actor 永远没有梯度。


## 6. PPO 在 RTX 3090 Ti 上的真实 GPU 路径
![PPO GPU 与三类 Loss 路径](/images/minimind-ppo-real-complete/04-ppo-gpu-loss-path.svg)


```text
CPU JSONL/template/tokenizer
→ input_ids [1,561] 搬到 CUDA
→ Actor no_grad rollout，completion [1,106]
→ old logps [1,106]
→ Reward Model FP16 forward，reward [1]
→ Critic no_grad forward，old values [1,106]
→ GPU Tensor 递推 TD/GAE，advantages/returns [1,106]
→ Current Actor + Current Critic forward（训练时保留图）
→ Actor clip + Reference KL + value loss
→ scalar [] → backward → AdamW Actor/Critic step
```

真实只读显存：Actor+Reference+Critic allocated 808.4 MiB；rollout peak 963.4 MiB；加载并运行 1.8B Reward Model 后 peak 4175.2 MiB。它不是训练峰值，因为 probe 没有梯度、反向激活和 AdamW 状态；allocated 也不等于 reserved 或 nvidia-smi 进程显存。

## 7. PPO 一句话总链路

```text
真实 5 条历史消息
→ prompt 991 chars
→ input_ids [1,561]
→ Actor completion [1,106]
→ output [1,667]
→ final reward 1.199707
→ token_rewards [0,...,0,1.199707]
→ Critic values [1,106]
→ TD + GAE
→ normalized advantages [1,106] 给 Actor
→ returns [1,106] 给 Critic
→ total loss 0.013138（更新前只读 forward）
```

<!-- ORIGINAL_NOTE_END -->
