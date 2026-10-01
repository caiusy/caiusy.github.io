---
title: 从 DeepSeek-V2 到 V4：MoE、注意力与 Loss 的完整演进
date: 2026-10-01 23:23:23
updated: 2026-10-01 23:23:23
mathjax: true
description: 从细粒度专家与 MLA，到无辅助损失负载均衡、MTP、CSA/HCA、mHC、Muon 和 OPD，以九张原创图、公式推导和完整数值例子拆解 DeepSeek V2/V3/V4 的创新与训练目标。
categories:
  - AI与大模型
  - 深度学习
tags:
  - DeepSeek
  - MoE
  - MLA
  - MTP
  - Sparse-Attention
  - GRPO
  - 知识蒸馏
type: deep-dive
difficulty: advanced
review_status: published
cover: /images/deepseek-v2-v3-v4-moe-attention-loss/01_evolution.png
---

从 V2 到 V4，DeepSeek 并非只是在增加专家数量：它逐步改变了参数如何分工、历史如何被读取，以及训练信号如何进入模型。本文用九张图，沿着“问题 → 机制 → 公式 → 数值例子 → 局限”的顺序，解释 MoE、MLA、MTP、CSA/HCA、mHC 与 Muon，并逐项拆开预训练、SFT、GRPO 和 OPD 的 loss。

<!-- more -->

> **阅读范围**：V2、V3 与 V4 预览版技术报告，不讨论 V4.1。所有配图为机制示意，收益注明比较条件。预训练、SFT、RL 与蒸馏分阶段讨论；教学例子的数值不代表实测日志。

## 导读：这条路线解决的不是一个瓶颈

设想你要部署一个既懂很多知识，又能读完一本长书的大模型。很快会遇到三张账单。

第一张是**参数计算账单**。Dense 模型处理每个 token 时，几乎每层的全部参数都要参与计算；模型越大，每一步越贵。MoE 把一层中的前馈网络拆成多个专家，只让一部分专家处理当前 token，从而把“模型能容纳多少知识”和“每次要做多少计算”部分分开。

第二张是**历史读取账单**。自回归生成时，新 token 要访问先前 token 的 key/value。即使 FFN 已经稀疏化，KV cache 仍会随着上下文增长。MLA 减少每个历史 token 要保存的数值；更进一步的压缩与稀疏注意力，减少历史条目的数量或每一步真正读取的条目数。

第三张是**训练组织账单**。专家选得太集中，GPU 会互相等待；为了强制均衡而添加的 loss，又可能妨碍专家专精。更多训练数据、更密集的监督、低精度计算和更好的分布式调度，必须一起工作，稀疏架构才会产生真实收益。

![三代技术演进](/images/deepseek-v2-v3-v4-moe-attention-loss/01_evolution.png)

*图 1：V2 建立稀疏计算与 KV 压缩的基础；V3 调整负载控制和预测目标；V4 把优化重点扩展到百万上下文、训练稳定性和多教师能力整合。*

| 配置 | V2 | V3 | V4-Flash | V4-Pro |
|---|---:|---:|---:|---:|
| 总参数 / 每 token 激活参数 | 236B / 21B | 671B / 37B | 284B / 13B | 1.6T / 49B |
| 主干层数 / 隐藏维度 | 60 / 5120 | 61 / 7168 | 43 / 4096 | 61 / 7168 |
| 共享专家 + 路由专家 | 2 + 160 | 1 + 256 | 1 + 256 | 1 + 384 |
| 每 token 选中的路由专家数 | 6 | 8 | 6 | 6 |
| 单专家中间维度 | 1536 | 2048 | 2048 | 3072 |
| 最前面的 FFN 层 | 第 1 层 dense | 前 3 层 dense | 前 3 层 Hash MoE | 前 3 层 Hash MoE |
| 注意力 | MLA | MLA | CSA + HCA | CSA + HCA |
| 预训练 token | 8.1T | 14.8T | 32T | 33T |
| 报告支持的上下文 | 128K | 128K | 1M | 1M |

这里的激活参数是计算规模指标，**不等于模型只需要装入这么多权重，也不直接等于显存或延迟**。专家权重存储、KV cache、激活、并行通信都会影响部署成本。[1][2][3]

## 一、先理解 MoE：为什么“更多参数”可以不意味着“同比例更多计算”

### 1.1 专家不是一整个聊天模型

在这类 Transformer 中，MoE 通常替换的是 FFN 子层。一个专家本质上是一个前馈网络，而不是一个能独立回答问题的模型。以 SwiGLU 为例，省略偏置：

{% raw %}
$$
E_i(u)=W_{i,\mathrm{down}}\left[\operatorname{SiLU}(W_{i,\mathrm{gate}}u)\odot(W_{i,\mathrm{up}}u)\right].
$$
{% endraw %}

若输入维度为 {% raw %}<span class="math-inline">$d$</span>{% endraw %}、专家中间维度为 {% raw %}<span class="math-inline">$m$</span>{% endraw %}，三块投影矩阵合计约有 {% raw %}<span class="math-inline">$3dm$</span>{% endraw %} 个参数。Top-{% raw %}<span class="math-inline">$K$</span>{% endraw %} 路由只执行被选中的 {% raw %}<span class="math-inline">$K$</span>{% endraw %} 个路由专家，但全部专家的权重仍要被保存、分片或加载。

把共享专家、路由专家与残差合起来，可以写成：

{% raw %}
$$
y_t=u_t+\sum_{j=1}^{N_s}E_j^{(s)}(u_t)
+\rho\sum_{i\in\mathcal S_t}g_{i,t}E_i^{(r)}(u_t).
$$
{% endraw %}

{% raw %}<span class="math-inline">$N_s$</span>{% endraw %} 是共享专家数，{% raw %}<span class="math-inline">$\mathcal S_t$</span>{% endraw %} 是该 token 选中的路由专家集合，{% raw %}<span class="math-inline">$g_{i,t}$</span>{% endraw %} 是路由权重；{% raw %}<span class="math-inline">$\rho$</span>{% endraw %} 表示实现可能使用的路由输出缩放。后续讨论权重时主要分析 {% raw %}<span class="math-inline">$g$</span>{% endraw %}，不把额外尺度与归一化混淆。

![共享专家与路由专家](/images/deepseek-v2-v3-v4-moe-attention-loss/02_moe.png)

*图 2：共享支路与条件支路同时工作。专家学到的分工由训练形成，不能简单预先标成“数学专家”“中文专家”。*

### 1.2 细粒度专家：改变可组合的计算单位

若把一个宽度为 {% raw %}<span class="math-inline">$m$</span>{% endraw %} 的大专家拆成若干宽度更小的专家，并相应增加每次激活的专家数，就能在相近的激活 FFN 宽度下提供更灵活的组合。

以 V2 为例，路由专家的激活中间宽度合计为 {% raw %}<span class="math-inline">$6\times1536=9216$</span>{% endraw %}；再加上两个共享专家的 {% raw %}<span class="math-inline">$2\times1536=3072$</span>{% endraw %}，总计为 12288。这个宽度账本解释了为什么它能提供很多候选专家，却不让每 token 的 FFN 计算按全部专家数增长。

但这只是 FFN 主计算的粗略比较：更多专家意味着更多路由、调度与小矩阵计算，实际效率要由实现验证。“组合数很大”也不是模型能力的数学证明。

### 1.3 共享专家：让公共模式少重复一些

如果所有专家都要分别学习语法、常见搭配、基础转换，就会浪费容量。共享专家的设计意图是让公共模式有一个稳定的计算通道，路由专家则更容易形成差异化。

这是一种**结构上的分工诱导**，不是把知识硬性切分后的保证。DeepSeekMoE 的细粒度专家与共享专家隔离早于 V2，已有独立论文；V2 的贡献是将其与 MLA 等设计结合并扩展到更大的语言模型。[4]

## 二、V2 的注意力创新：MLA 为什么能显著缩小 KV cache？

### 2.1 普通多头注意力的缓存为什么贵

对于每个历史位置，MHA 要保存每个头的 key 与 value。假设有 {% raw %}<span class="math-inline">$n_h$</span>{% endraw %} 个头，每头 K、V 都为 {% raw %}<span class="math-inline">$d_h$</span>{% endraw %} 维，则每层每个历史 token 要缓存约：

{% raw %}
$$
2n_hd_h
$$
{% endraw %}

个数值。总缓存随层数、历史长度、并发序列数和每个数值的字节数共同增长。低并发时可接受的缓存，放到长上下文和大批量服务中，就可能成为显存瓶颈。

### 2.2 MLA 的核心：K 与 V 共享一个低维来源

MLA，Multi-head Latent Attention，不再把完整 K、V 都作为必须保存的状态，而是让它们来自同一个压缩向量：

{% raw %}
$$
c_t^{KV}=W^{DKV}h_t,\qquad
k_{t,i}^{C}=W_i^{UK}c_t^{KV},\qquad
v_{t,i}^{C}=W_i^{UV}c_t^{KV}.
$$
{% endraw %}

{% raw %}<span class="math-inline">$h_t\in\mathbb R^d$</span>{% endraw %} 是输入，{% raw %}<span class="math-inline">$c_t^{KV}\in\mathbb R^{d_c}$</span>{% endraw %} 是联合 KV 潜变量。直觉上，多个头的 K/V 不再各自存一份完整表示，而是共享一份低维底稿，通过各头投影解释它。

Query 也采用低秩投影，但 Query 不需要像历史 K/V 一样跨步缓存。因此，**Query 压缩与 KV cache 压缩不能混为同一笔节省**。

### 2.3 为什么必须解耦 RoPE

内容相关性和位置相关性分别计算：

{% raw %}
$$
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}],\qquad
k_{j,i}=[k_{j,i}^{C};k_j^{R}],
$$
{% endraw %}

{% raw %}
$$
q_{t,i}^{R}=\operatorname{RoPE}_t(\widetilde q_{t,i}^{R}),\qquad
k_j^{R}=\operatorname{RoPE}_j(W^{KR}h_j).
$$
{% endraw %}

于是注意力打分为：

{% raw %}
$$
a_{t,j,i}=\operatorname{softmax}_{j\le t}
\left(\frac{(q_{t,i}^{C})^Tk_{j,i}^{C}+(q_{t,i}^{R})^Tk_j^{R}}
{\sqrt{d_h+d_h^R}}\right).
$$
{% endraw %}

位置 key 在头之间共享。关键在于，内容分支可以改写为：

{% raw %}
$$
(q_{t,i}^{C})^TW_i^{UK}c_j^{KV}
=\left((W_i^{UK})^Tq_{t,i}^{C}\right)^Tc_j^{KV}.
$$
{% endraw %}

这样就不必先为所有历史位置恢复完整 key。Value 分支也可利用线性性，在潜空间聚合之后再投影，并与后续输出投影结合。

如果把依赖历史位置的旋转直接混入内容投影，一般就不能如此简单地把它吸收到一个与历史位置无关的固定变换中。因此 MLA 单独保存位置 key。这里的收益是**避免显式缓存或恢复全部多头 K/V**，并不是“所有重建和注意力计算都免费”。

### 2.4 用一个口径清楚的例子算账

V2 的相关维度为 {% raw %}<span class="math-inline">$n_h=128$</span>{% endraw %}、{% raw %}<span class="math-inline">$d_h=128$</span>{% endraw %}、{% raw %}<span class="math-inline">$d_c=512$</span>{% endraw %}、{% raw %}<span class="math-inline">$d_h^R=64$</span>{% endraw %}。若比较一个 K/V 各 128 维、128 头的 MHA 基线：

{% raw %}
$$
\frac{2\times128\times128}{512+64}\approx56.9.
$$
{% endraw %}

在 BF16、只统计这一层的 KV 数据时，基线为 64 KiB/token，MLA 为 1.125 KiB/token。

![MLA 缓存比较](/images/deepseek-v2-v3-v4-moe-attention-loss/03_mla.png)

*图 3：约 57 倍是上述指定维度的缓存数值量比较；V2 报告中相对 DeepSeek 67B 的“KV cache 减少 93.3%”采用另一基线。二者不能互换，更不能直接当作端到端提速倍数。*

MLA 主要压缩的是**每个历史位置的表示维度**。它没有自动解决“每次注意力仍需访问很长历史”的全部问题，这正是后续稀疏化与序列压缩要处理的部分。[1, §2.1]

## 三、V2 的三个均衡 loss：不是重复罚三次，而是约束三个瓶颈

### 3.1 主任务：交叉熵到底优化什么

给定序列 {% raw %}<span class="math-inline">$x_1,\ldots,x_T$</span>{% endraw %}，语言建模目标为：

{% raw %}
$$
L_{\mathrm{LM}}=-\frac1{T-1}\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\le t}).
$$
{% endraw %}

它提高真实下一个 token 的概率。真实 token 的预测概率从 0.1 提升到 0.5，该位置的损失就从 {% raw %}<span class="math-inline">$-\log0.1\approx2.303$</span>{% endraw %} 降至 {% raw %}<span class="math-inline">$-\log0.5\approx0.693$</span>{% endraw %}。

对词表 logit {% raw %}<span class="math-inline">$z_v$</span>{% endraw %}，单位置交叉熵的梯度为：

{% raw %}
$$
\frac{\partial\ell}{\partial z_v}=p_v-\mathbf1[v=x_{t+1}].
$$
{% endraw %}

这个目标关心预测质量，却不会主动保证所有 GPU 一样忙。路由若早期偏向少数专家，这些专家获得更多训练机会，可能形成进一步集中的反馈。

### 3.2 路由得分与统计口径

V2 用 softmax 产生专家亲和度：

{% raw %}
$$
s_{i,t}=\frac{\exp(e_i^Tu_t)}{\sum_{j=1}^{N_r}\exp(e_j^Tu_t)}.
$$
{% endraw %}

在设备限制下选择 {% raw %}<span class="math-inline">$K_r$</span>{% endraw %} 个路由专家。令 {% raw %}<span class="math-inline">$a_{i,t}$</span>{% endraw %} 表示专家 {% raw %}<span class="math-inline">$i$</span>{% endraw %} 是否被 token {% raw %}<span class="math-inline">$t$</span>{% endraw %} 选中。对一条包含 {% raw %}<span class="math-inline">$T$</span>{% endraw %} 个 token 的序列定义：

{% raw %}
$$
f_i=\frac{N_r}{K_rT}\sum_{t=1}^{T}a_{i,t},\qquad
P_i=\frac1T\sum_{t=1}^{T}s_{i,t}.
$$
{% endraw %}

{% raw %}<span class="math-inline">$f_i$</span>{% endraw %} 是归一化的实际选中频率；完全平均时 {% raw %}<span class="math-inline">$f_i=1$</span>{% endraw %}。{% raw %}<span class="math-inline">$P_i$</span>{% endraw %} 是平均软概率，完全平均时 {% raw %}<span class="math-inline">$P_i=1/N_r$</span>{% endraw %}。

后文的 {% raw %}<span class="math-inline">$L_{\mathrm{exp}}$</span>{% endraw %}、{% raw %}<span class="math-inline">$L_{\mathrm{dev}}$</span>{% endraw %}、{% raw %}<span class="math-inline">$L_{\mathrm{com}}$</span>{% endraw %} 均定义成**不含系数**的损失。原报告把系数放在各项内部，本文把它们提到总式外，避免重复乘系数。

### 3.3 专家级：抑制路由塌缩

{% raw %}
$$
L_{\mathrm{exp}}=\sum_{i=1}^{N_r}f_iP_i.
$$
{% endraw %}

可以把它理解成：哪个专家实际已经很忙，就对继续把软概率分给它施加更高代价。

由于 Top-K 的离散选择通常不参与直接求导，反向传播把 {% raw %}<span class="math-inline">$f_i$</span>{% endraw %} 当作统计常量，经 {% raw %}<span class="math-inline">$P_i$</span>{% endraw %} 回传。对于某个 token 的路由 logit {% raw %}<span class="math-inline">$z_{j,t}$</span>{% endraw %}，有：

{% raw %}
$$
\frac{\partial L_{\mathrm{exp}}}{\partial z_{j,t}}
=\frac1T s_{j,t}\left(f_j-\sum_i f_is_{i,t}\right).
$$
{% endraw %}

因此，一个负载高于该 token 概率加权平均负载的专家，会得到压低其 logit 的梯度。这比“惩罚不均衡”一句话更准确地说明了 loss 如何起作用。

**小例子。** 设 {% raw %}<span class="math-inline">$N_r=4,K_r=1,T=8$</span>{% endraw %}。若选中次数为 {% raw %}<span class="math-inline">$(2,2,2,2)$</span>{% endraw %}，则 {% raw %}<span class="math-inline">$f=(1,1,1,1)$</span>{% endraw %}，均匀 {% raw %}<span class="math-inline">$P=(0.25,0.25,0.25,0.25)$</span>{% endraw %} 对应 {% raw %}<span class="math-inline">$L=1$</span>{% endraw %}。若选中次数为 {% raw %}<span class="math-inline">$(8,0,0,0)$</span>{% endraw %}，且 {% raw %}<span class="math-inline">$P=(0.7,0.1,0.1,0.1)$</span>{% endraw %}，则 {% raw %}<span class="math-inline">$f=(4,0,0,0)$</span>{% endraw %}，损失为 2.8。

这里的 1 是均衡参考值，不应把这个乘积型代理目标说成“严格的负载方差”或“所有情况下以 1 为全局下界”。硬路由与软概率的关系、Top-K 和统计样本都会影响它。

### 3.4 设备级：避免有的 GPU 计算拥堵

设共有 {% raw %}<span class="math-inline">$D$</span>{% endraw %} 台设备，设备 {% raw %}<span class="math-inline">$d$</span>{% endraw %} 上的专家集合为 {% raw %}<span class="math-inline">$\mathcal E_d$</span>{% endraw %}，则：

{% raw %}
$$
f_d^{\mathrm{dev}}=\frac1{|\mathcal E_d|}\sum_{i\in\mathcal E_d}f_i,
\qquad P_d=\sum_{i\in\mathcal E_d}P_i,
$$
{% endraw %}

{% raw %}
$$
L_{\mathrm{dev}}=\sum_{d=1}^{D}f_d^{\mathrm{dev}}P_d.
$$
{% endraw %}

专家并行的训练步经常要等最慢设备完成。因而设备总负载不均，会直接造成其他 GPU 空等。严格均匀的专家负载当然会带来设备均衡，但实践中的均衡是软约束；单独设置设备级目标，可以优先压制影响吞吐的设备拥堵，而不必把每个专家都强迫到完全相同的负载。

### 3.5 通信级：计算均衡，不代表收到的 token 数均衡

令 {% raw %}<span class="math-inline">$r_{d,t}=1$</span>{% endraw %} 表示 token {% raw %}<span class="math-inline">$t$</span>{% endraw %} 被发往设备 {% raw %}<span class="math-inline">$d$</span>{% endraw %}，每个 token 最多发往 {% raw %}<span class="math-inline">$M$</span>{% endraw %} 台设备：

{% raw %}
$$
f_d^{\mathrm{com}}=\frac{D}{MT}\sum_t r_{d,t},\qquad
L_{\mathrm{com}}=\sum_d f_d^{\mathrm{com}}P_d.
$$
{% endraw %}

为什么还需要这一项？同一个 token 在一台设备上可能使用多个专家。设备执行的专家次数，与需要跨设备传输的不同 token 数，并不是同一个统计量。

例如，两台设备都执行了 100 次专家计算：一台来自 50 个 token、每个 token 使用两个本地专家；另一台来自 100 个 token、每个只使用一个。计算量类似，接收 token 的通信负载却不同。

V2 的 device-limited routing 先限制一个 token 最多去多少台设备，通信 loss 再缓解接收侧的热点。其大模型配置为 {% raw %}<span class="math-inline">$D=8,M=3$</span>{% endraw %}。注意这里是**设备**，不能随意换成服务器节点。[1, §2.2–3.1]

### 3.6 V2 的总式与代价

{% raw %}
$$
\boxed{L_{\mathrm{V2,pre}}=L_{\mathrm{LM}}+
0.003L_{\mathrm{exp}}+0.05L_{\mathrm{dev}}+0.02L_{\mathrm{com}}.}
$$
{% endraw %}

此式是对语言模型主目标和报告均衡项的汇总；跨层、跨序列的聚合须保持与实现一致。不能仅凭 0.05 大于 0.003，就断言某项的梯度必然大多少倍。

这些目标有正当用途，但也带来权衡：一个 token 可能最适合某专家，均衡目标却推动它去别处。**不是所有辅助梯度都“污染”训练，而是均衡和任务质量之间存在需要调节的冲突。** V2 还使用训练时 token-dropping 控制设备预算，评估时不丢弃；这里的“丢弃”是跳过部分专家处理，不是把整个训练序列从数据集删除。

## 四、V3 的路由创新：把负载控制从主梯度里解耦出来

### 4.1 选择和加权，原来可以用不同的分数

V3 首先以 sigmoid 计算正的亲和度：

{% raw %}
$$
s_{i,t}=\sigma(e_i^Tu_t).
$$
{% endraw %}

然后用“亲和度 + 负载偏置”选择专家：

{% raw %}
$$
\mathcal S_t=\operatorname{TopKIndices}_i(s_{i,t}+b_i,K_r).
$$
{% endraw %}

真正混合专家输出时，仍使用**未加偏置的亲和度**：

{% raw %}
$$
g_{i,t}=\frac{\mathbf1[i\in\mathcal S_t]s_{i,t}}
{\sum_{j\in\mathcal S_t}s_{j,t}}.
$$
{% endraw %}

因此 {% raw %}<span class="math-inline">$b_i$</span>{% endraw %} 改变“谁有机会工作”，不直接变成“这个专家的语义贡献权重”。Sigmoid 的得分不在全部专家间和为 1，但 Top-K 后会重新归一化；这属于设计选择，不能仅凭“未归一化”就列为缺点。

![V3 路由数值例子](/images/deepseek-v2-v3-v4-moe-attention-loss/04_router.png)

*图 4：一个原本分数低的欠载专家可以被偏置选进来，但输出仍按原始亲和度加权。选中集合内重新归一化后，E2、E3 的权重分别约为 0.778、0.222。*

### 4.2 偏置是反馈控制器，不是额外的可微正则项

设 {% raw %}<span class="math-inline">$n_i$</span>{% endraw %} 为一个训练步内分配给专家 {% raw %}<span class="math-inline">$i$</span>{% endraw %} 的负载，{% raw %}<span class="math-inline">$\bar n$</span>{% endraw %} 为目标平均负载，控制规则可概括为：

{% raw %}
$$
b_i\leftarrow b_i+\gamma\operatorname{sign}(\bar n-n_i).
$$
{% endraw %}

过载就降低偏置，欠载就提高偏置。这个更新依据负载统计进行，不通过语言建模 loss 对 {% raw %}<span class="math-inline">$b_i$</span>{% endraw %} 求梯度。

这意味着训练同时存在两种变化：

- 模型参数通过任务与辅助预测目标的梯度更新；
- 路由偏置通过负载反馈调整。

二者并非互不影响：偏置改变了哪些专家接收 token，因此间接影响后续训练数据分配。但它避免了用较强的全局均衡 loss 直接改写路由梯度。

V3 报告给出的 {% raw %}<span class="math-inline">$\gamma$</span>{% endraw %} 在前 14.3T token 为 0.001，最后 500B token 为 0，即末段冻结这种偏置更新。这意味着 0.001 不是全程不变的设置。[2, §4.2]

### 4.3 为什么还保留序列级均衡 loss

批次整体均衡，不代表每条序列都不会极端集中；而强迫每条序列均衡，又可能削弱自然的领域专精。V3 选择的是**全局以偏置调节为主，单序列用极轻量目标防极端情况**。

按照 V3 报告式 (18)–(20)，令：

{% raw %}
$$
\widetilde s_{i,t}=\frac{s_{i,t}}{\sum_j s_{j,t}},\quad
P_i=\frac1T\sum_t\widetilde s_{i,t},
$$
{% endraw %}

{% raw %}
$$
f_i=\frac{N_r}{K_rT}\sum_t\mathbf1\!\left[i\in\operatorname{TopKIndices}_j(s_{j,t},K_r)\right],
\qquad L_{\mathrm{seq}}=\sum_i f_iP_i.
$$
{% endraw %}

一个容易漏掉的细节：**报告这里的 Top-K 按原始 {% raw %}<span class="math-inline">$s$</span>{% endraw %} 写，实际专家选择按 {% raw %}<span class="math-inline">$s+b$</span>{% endraw %} 写**。讲解公式时应保留这种区别，不能未经说明就把两处的统计集合改成同一个。

V3 的系数是 {% raw %}<span class="math-inline">$\alpha=10^{-4}$</span>{% endraw %}。所以 auxiliary-loss-free 指的是主要负载均衡策略，而不是“模型训练没有任何辅助 loss”。MTP 也是辅助训练目标，但并非负载均衡 loss。

V3 还限制每个 token 最多路由到 4 个节点，并通过负载和部署策略实现训练与推理不丢 token。这些都是“专精、均衡、通信”共同设计的结果，不应全归因于 sigmoid。[2, §2.1.2]

### 4.4 把一个 token 走完：路由、聚合与 loss 的教学例子

把真实模型缩小成二维、3 个候选专家、Top-2 的玩具例子，下面的数值全部人为设定，用来串起计算流程，不是 V3 运行日志。

输入 {% raw %}<span class="math-inline">$u=[1,0]$</span>{% endraw %}。设原始亲和度为 {% raw %}<span class="math-inline">$(0.8,0.7,0.2)$</span>{% endraw %}，偏置为 {% raw %}<span class="math-inline">$(-0.3,0,0.4)$</span>{% endraw %}，选择分就变为 {% raw %}<span class="math-inline">$(0.5,0.7,0.6)$</span>{% endraw %}，因此选择 E2、E3。加权仍取原始亲和度：

{% raw %}
$$
g_2=\frac{0.7}{0.7+0.2}=\frac79,\qquad g_3=\frac29.
$$
{% endraw %}

假设共享专家输出 {% raw %}<span class="math-inline">$E_s(u)=[1,1]$</span>{% endraw %}，两个选中专家输出分别为 {% raw %}<span class="math-inline">$E_2(u)=[2,0]$</span>{% endraw %}、{% raw %}<span class="math-inline">$E_3(u)=[0,3]$</span>{% endraw %}，路由输出尺度暂设为 1。聚合后：

{% raw %}
$$
y=[1,0]+[1,1]+\frac79[2,0]+\frac29[0,3]
=\left[\frac{32}{9},\frac53\right]\approx[3.556,1.667].
$$
{% endraw %}

后续层与输出头把这个表示继续变换为词表分布。为说明 loss 汇总，再假设主头给真实下一 token 的概率为 0.5，MTP 给其真实目标的概率为 0.25，序列均衡统计得到 {% raw %}<span class="math-inline">$L_{\mathrm{seq}}=1.2$</span>{% endraw %}。使用 {% raw %}<span class="math-inline">$\lambda=0.3$</span>{% endraw %}：

{% raw %}
$$
\begin{aligned}
L&=-\log0.5+0.3(-\log0.25)+10^{-4}\times1.2\\
&\approx0.693147+0.415888+0.000120\\
&=1.109155.
\end{aligned}
$$
{% endraw %}

这一步的梯度会经输出头、后续层回到主干及被执行的专家；MTP 也通过自己的因果模块回传。未选中专家不会获得该 token 的专家输出支路梯度，路由打分则还受归一化与序列均衡项影响。偏置的下一次变化不从 1.109155 求导，而是看训练步负载：过载专家减偏置，欠载专家加偏置。

这个例子把两条通道完整分开了：**数值预测误差驱动参数学习，负载统计驱动选择机会调节。**

## 五、MTP：多预测一个 token，究竟多了什么监督？

### 5.1 不是简单地在同一 hidden state 上接两个独立输出头

普通 next-token prediction 从 {% raw %}<span class="math-inline">$x_{\le t}$</span>{% endraw %} 的表示预测 {% raw %}<span class="math-inline">$x_{t+1}$</span>{% endraw %}。DeepSeek 的 MTP，Multi-Token Prediction，进一步把主干表示与后续已知 token 的嵌入合并，按深度串行预测。

设 {% raw %}<span class="math-inline">$h_t^{(0)}$</span>{% endraw %} 是主模型输出，第 {% raw %}<span class="math-inline">$k$</span>{% endraw %} 个 MTP 模块为：

{% raw %}
$$
\widetilde h_t^{(k)}=M_k\left[
\operatorname{RMSNorm}(h_t^{(k-1)});
\operatorname{RMSNorm}(\operatorname{Emb}(x_{t+k}))\right],
$$
{% endraw %}

{% raw %}
$$
h_{1:T-k}^{(k)}=\operatorname{TRM}_k(\widetilde h_{1:T-k}^{(k)}),\qquad
p_t^{(k)}=\operatorname{Softmax}(\operatorname{OutHead}(h_t^{(k)})).
$$
{% endraw %}

{% raw %}<span class="math-inline">$M_k$</span>{% endraw %} 把两个 {% raw %}<span class="math-inline">$d$</span>{% endraw %} 维向量拼接后的 {% raw %}<span class="math-inline">$2d$</span>{% endraw %} 维压回 {% raw %}<span class="math-inline">$d$</span>{% endraw %} 维；{% raw %}<span class="math-inline">$\operatorname{TRM}_k$</span>{% endraw %} 是因果 Transformer 模块。Embedding 和输出头与主模型共享。第 {% raw %}<span class="math-inline">$k$</span>{% endraw %} 个模块在位置 {% raw %}<span class="math-inline">$t$</span>{% endraw %} 的目标是 {% raw %}<span class="math-inline">$x_{t+k+1}$</span>{% endraw %}。

![MTP 因果链](/images/deepseek-v2-v3-v4-moe-attention-loss/06_mtp.png)

*图 5：V3 与 V4 的 MTP 深度均为 1，即主模型预测下一 token，额外模块再向前预测一步。*

### 5.2 为什么输入真实的下一 token 不算作弊

假设主干已经看到“今天 / 天气”。主输出头预测下一个 token“很好”；MTP 在训练中接收“很好”的嵌入，再预测它后面的 token。

MTP 接触到了自己目标之前的真实 token，这是 teacher forcing；它没有接触自己正在预测的目标。关键是因果 mask 与位置对齐正确，不能让目标或之后的信息反向泄露。

因此，“MTP 一次凭空预测多个未来 token”是误导。训练中的额外预测有完整的条件链；推理时若没有真实 token，就必须使用生成的候选并做相应验证。

### 5.3 损失如何计算，梯度流向哪里

令额外预测深度为 {% raw %}<span class="math-inline">$D_{\mathrm{MTP}}$</span>{% endraw %}，以有效位置数归一化的讲解写法为：

{% raw %}
$$
L_{\mathrm{MTP}}^{(k)}=-\frac1{T-k-1}
\sum_{t=1}^{T-k-1}\log p_t^{(k)}[x_{t+k+1}],
$$
{% endraw %}

{% raw %}
$$
L_{\mathrm{MTP}}=\frac1{D_{\mathrm{MTP}}}\sum_{k=1}^{D_{\mathrm{MTP}}}L_{\mathrm{MTP}}^{(k)}.
$$
{% endraw %}

这里 {% raw %}<span class="math-inline">$L_{\mathrm{MTP}}$</span>{% endraw %} **不含权重 {% raw %}<span class="math-inline">$\lambda$</span>{% endraw %}**。V3 原报告使用固定 {% raw %}<span class="math-inline">$T$</span>{% endraw %} 的归一化和相应移位索引，且把 {% raw %}<span class="math-inline">$\lambda$</span>{% endraw %} 放在最终 MTP 定义中；实现时应选择一种口径并保持一致，不能重复加权。

MTP 梯度不仅训练新增模块，也通过 {% raw %}<span class="math-inline">$h_t^{(0)}$</span>{% endraw %} 回到主干，并更新共享嵌入和输出头。主干因此被要求形成对后续预测仍有用的表示。这就是“训练信号更密”的具体含义。

V3 的总目标为：

{% raw %}
$$
\boxed{L_{\mathrm{V3,pre}}=L_{\mathrm{LM}}+
\lambda L_{\mathrm{MTP}}+10^{-4}L_{\mathrm{seq}}.}
$$
{% endraw %}

前 10T token 使用 {% raw %}<span class="math-inline">$\lambda=0.3$</span>{% endraw %}，后 4.8T 使用 0.1。较小的末段权重降低辅助目标的相对影响；这并不代表 MTP 在训练末段被删除。

### 5.4 推理时为什么既可以删除，也可以用来加速

MTP 的训练收益已进入主干参数，因此推理时可以删除额外模块，主模型照常工作。也可以把 MTP 当作候选生成器，配合主模型验证进行投机解码。

V3 报告给出约 85%–90% 的第二 token 接受率与约 1.8 倍 TPS 提升。这是其评估与部署条件下的结果，依赖接受率、验证成本、batch、硬件和生成策略，不能当作任何服务都能获得的固定倍数。[2, §2.2、§5.4.3]

## 六、V3 的工程创新：让稀疏架构真的跑得便宜

### 6.1 FP8 的关键不是“把全部浮点数改成 8 位”

V3 在主要 GEMM 中使用 FP8，但路由、归一化、注意力等敏感部分，以及主权重、梯度和优化器状态中的关键部分保留更高精度。

对一个量化块，可用以下示意式理解：

{% raw %}
$$
\widehat X=a\,Q_{\mathrm{FP8}}(X/a).
$$
{% endraw %}

{% raw %}<span class="math-inline">$a$</span>{% endraw %} 是该块的尺度。若整张矩阵共用一个尺度，一个异常大值就可能迫使其他数值挤进很粗的刻度；分块尺度缩小了异常值的影响范围。V3 使用激活的细粒度分组与权重块量化，并周期性把部分累加结果提升到更高精度，避免长内积的误差持续积累。

这解决的是**数值表示与计算误差**，不是新增了一个“FP8 loss”。报告中 FP8/BF16 的小规模对照得到小于 0.25% 的相对 loss 差异，也不等于所有权重或任务指标误差都小于 0.25%。[2, §3.3、附录 B]

### 6.2 DualPipe：把等待网络的时间藏进计算里

专家并行意味着 token 要跨设备分发，再收回专家输出。DualPipe 通过安排前向、反向与流水线通信的重叠，减少流水线气泡，让通信与可执行的计算同时发生。

“通信被隐藏”不代表通信消失：它依赖可重叠的工作量、网络带宽和调度。V3 不使用张量并行，是模型配置、内存优化、流水线、专家并行等多项设计共同支持的结果，不能说是 DualPipe 单独带来的普适结论。

报告给出的 2.788M H800 GPU 小时，按 2 美元/GPU 小时估算为 557.6 万美元，覆盖其列示的正式训练阶段；**不包含此前研究、架构与数据消融的全部成本，也不是公司的总研发成本**。[2, 表 1]

## 七、V4：从压缩每个 KV，走向压缩历史本身

### 7.1 中间不能漏掉 DSA 这座桥

V3 之后的 DeepSeek-V3.2 系列引入 DSA，DeepSeek Sparse Attention：先用较轻的 lightning indexer 判断值得读取的历史位置，再对选中的位置做主注意力。[5]

V4 在此基础上继续前进：**先把历史压缩成较少条目，再选择性读取；另一类层则把历史压得更粗，然后整体读取。** 两种层交替构成混合注意力。具体到前两层，Flash 使用纯滑动窗口注意力，Pro 使用 HCA；此后再采用 CSA/HCA 交替配置。

“MLA 被替换”应理解为原有的 MLA 注意力架构被新的 CSA/HCA 方案替代，不代表低秩 Query 等思想完全消失。V4 仍使用低秩 Query 投影。

### 7.2 CSA：学习压缩，而不是简单平均四个 token

CSA，Compressed Sparse Attention，先生成 KV 候选 {% raw %}<span class="math-inline">$C^a,C^b$</span>{% endraw %} 与逐通道压缩 logit {% raw %}<span class="math-inline">$Z^a,Z^b$</span>{% endraw %}。每个压缩条目汇集当前块和前一个块的候选，并使用可学习位置偏置：

{% raw %}
$$
S_i=\operatorname{Softmax}_{\mathrm{positions}}
\left([Z^a_{\mathrm{current}}+B^a;Z^b_{\mathrm{previous}}+B^b]\right),
$$
{% endraw %}

{% raw %}
$$
\overline C_i=\sum_{j\in\mathrm{current}}S^a_j\odot C^a_j
+\sum_{j\in\mathrm{previous}}S^b_j\odot C^b_j.
$$
{% endraw %}

这里每个通道都能学自己的位置权重，{% raw %}<span class="math-inline">$\odot$</span>{% endraw %} 是逐元素乘法。压缩步长为 {% raw %}<span class="math-inline">$m=4$</span>{% endraw %}，每个条目实际涉及两块、共 {% raw %}<span class="math-inline">$2m$</span>{% endraw %} 个候选；相邻条目覆盖区间重叠，输出条目数约为原来的 {% raw %}<span class="math-inline">$1/m$</span>{% endraw %}。

因此“4 合 1”只描述长度压缩率，不能据此画成无重叠的四个向量简单平均。重叠让块边界附近的信息有机会进入相邻摘要，但仍是有损压缩，不能保证细节全部保留。

### 7.3 Lightning indexer：先打便宜的分，再做昂贵的注意力

对当前位置 {% raw %}<span class="math-inline">$t$</span>{% endraw %} 和压缩历史条目 {% raw %}<span class="math-inline">$s$</span>{% endraw %}，索引分数为：

{% raw %}
$$
I_{t,s}=\sum_{h=1}^{n_h^I}w_{t,h}^{I}
\operatorname{ReLU}\left((q_{t,h}^{I})^Tk_s^{I,\mathrm{comp}}\right).
$$
{% endraw %}

再选择：

{% raw %}
$$
\mathcal C_t=\operatorname{TopKIndices}_s(I_{t,s},k).
$$
{% endraw %}

Flash 的 {% raw %}<span class="math-inline">$k=512$</span>{% endraw %}，Pro 的 {% raw %}<span class="math-inline">$k=1024$</span>{% endraw %}。这些数是**注意力选择的压缩条目数**，与 MoE 的 Top-6 专家不是同一件事。

主注意力采用共享 KV 的 MQA 形式，一个压缩条目同时充当 key 和 value，并在查询头之间共享。若暂时省略 sink，主计算可以示意为：

{% raw %}
$$
o_{t,h}=\sum_{s\in\mathcal C_t\cup\mathcal W_t}
\operatorname{softmax}_s(q_{t,h}^Tk_s/\sqrt c)\,v_s,
$$
{% endraw %}

其中 {% raw %}<span class="math-inline">$\mathcal W_t$</span>{% endraw %} 是局部未压缩窗口，{% raw %}<span class="math-inline">$|\mathcal W_t|=128$</span>{% endraw %}。窗口保留最近细节，压缩条目提供远处记忆；不是整段历史都只有压缩版本可访问。

### 7.4 HCA：把历史压得更粗，但不再做稀疏选择

HCA，Heavily Compressed Attention，压缩率为 {% raw %}<span class="math-inline">$m'=128$</span>{% endraw %}，使用非重叠块：

{% raw %}
$$
S_{j,r}=\frac{\exp(Z_{j,r}+B_{j\bmod m',r})}
{\sum_{q\in\mathcal B_i}\exp(Z_{q,r}+B_{q\bmod m',r})},
\qquad
\overline C_{i,r}=\sum_{j\in\mathcal B_i}S_{j,r}C_{j,r}.
$$
{% endraw %}

{% raw %}<span class="math-inline">$r$</span>{% endraw %} 是通道索引，{% raw %}<span class="math-inline">$\mathcal B_i$</span>{% endraw %} 是第 {% raw %}<span class="math-inline">$i$</span>{% endraw %} 个块。HCA 对全部可见的压缩条目做注意力，同时保留局部窗口。

以 1,048,576 个历史 token 为例，忽略边界状态：CSA 约产生 262,144 个压缩条目，主注意力再从中选 512 或 1024 个；HCA 产生 8192 个条目，全部读取。

![CSA 与 HCA](/images/deepseek-v2-v3-v4-moe-attention-loss/07_attention.png)

*图 6：CSA 提供较细粒度的选择性远程读取，HCA 提供更粗粒度的全局覆盖。它们是在不同层交替使用，不是所有层都同时跑一套完整 CSA 和 HCA。*

### 7.5 复杂度：主注意力稀疏，不等于整个模块恒定成本

CSA 的主注意力每个 query 约访问 {% raw %}<span class="math-inline">$k+w$</span>{% endraw %} 个条目，但 indexer 仍要为约 {% raw %}<span class="math-inline">$T/m$</span>{% endraw %} 个压缩候选打分，还要付出压缩、投影、Top-K 和缓存维护的成本。

HCA 每个 query 访问约 {% raw %}<span class="math-inline">$T/m'+w$</span>{% endraw %} 个条目。在固定压缩率下，其全序列注意力的渐近量级仍可写成 {% raw %}<span class="math-inline">$O(T^2/m')$</span>{% endraw %}；压缩大幅减小常数，却不应被宣传成严格线性注意力。

V4 报告在 1M 上下文条件下估算：相对 V3.2，Pro 的单 token 等效 FP8 FLOPs 约为 27%、KV cache 约为 10%；Flash 分别约为 10% 和 7%。这些收益合并了架构、稀疏度和低精度存储因素，**不是同硬件端到端延迟的直接测量倍数**。[3, 图 1、§2.3.4]

### 7.6 配套设计分别解决什么

- **Query / KV 的 RMSNorm**：控制数值尺度，降低 attention logit 爆炸风险。不能把它简化成严格余弦注意力，也不是附加 loss。
- **Partial RoPE**：在部分维度施加旋转位置编码，使位置信息与其他内容维度共同工作。
- **Grouped output projection**：先分组压缩各头输出，再汇总，降低大头维度带来的输出投影开销。
- **Attention sink**：为每个头增加可学习的 sink logit {% raw %}<span class="math-inline">$z'_h$</span>{% endraw %}，把 {% raw %}<span class="math-inline">$\exp(z'_h)$</span>{% endraw %} 放入 softmax 分母：

{% raw %}
$$
a_{h,t,j}=\frac{\exp(z_{h,t,j})}{\sum_k\exp(z_{h,t,k})+\exp(z'_h)}.
$$
{% endraw %}

这样真实条目的权重和可以小于 1；当所有条目都不值得强读时，模型可以减少这一头输出的总幅度。这里是可学习的“空吸收”机制，不一定是把一个普通文本 token 强行插到序列前面。

## 八、V4 的残差与优化器：扩大表达力，也控制数值风险

### 8.1 mHC 扩展的是残差流，而非把每个 FFN 都加宽四倍

普通残差形式为 {% raw %}<span class="math-inline">$x_{l+1}=x_l+F_l(x_l)$</span>{% endraw %}。mHC 把状态拓展成 {% raw %}<span class="math-inline">$n_{\mathrm{hc}}=4$</span>{% endraw %} 路：

{% raw %}
$$
X_l\in\mathbb R^{4\times d},\qquad
X_{l+1}=B_lX_l+C_lF_l(A_lX_l).
$$
{% endraw %}

其中 {% raw %}<span class="math-inline">$A_l\in\mathbb R^{1\times4}$</span>{% endraw %} 把四路状态汇聚给主模块，{% raw %}<span class="math-inline">$C_l\in\mathbb R^{4\times1}$</span>{% endraw %} 把主模块输出分发回四路，{% raw %}<span class="math-inline">$B_l\in\mathbb R^{4\times4}$</span>{% endraw %} 负责残差支路的混合。主模块输入输出仍为 {% raw %}<span class="math-inline">$d$</span>{% endraw %} 维。

![mHC 残差结构](/images/deepseek-v2-v3-v4-moe-attention-loss/08_mhc.png)

*图 7：多路状态增加了跨层信息传递的自由度。计算与显存也会增加，所以 V4 同时优化其内核、重计算与状态存储。*

### 8.2 双随机约束为什么有助于稳定

若 {% raw %}<span class="math-inline">$B_l$</span>{% endraw %} 是任意矩阵，多层连乘可能放大或衰减某些方向。mHC 要求它非负、每行每列之和均为 1：

{% raw %}
$$
B_l\mathbf1=\mathbf1,\qquad B_l^T\mathbf1=\mathbf1,\qquad B_l\ge0.
$$
{% endraw %}

这类矩阵组成 Birkhoff 多面体。对固定的、满足约束的 {% raw %}<span class="math-inline">$B_l$</span>{% endraw %}，有：

{% raw %}
$$
\|B_l\|_2\le\sqrt{\|B_l\|_1\|B_l\|_\infty}=1.
$$
{% endraw %}

所以残差混合本身不会任意放大欧氏范数。实现通过对原始参数取指数并做 Sinkhorn–Knopp 行列归一化逼近约束；V4 采用 20 次迭代。输入与输出映射分别以 sigmoid 与两倍 sigmoid 限定范围。

但这**不是整个网络梯度永不爆炸的证明**：完整更新还含 {% raw %}<span class="math-inline">$F_l$</span>{% endraw %}、输入相关映射的导数，以及有限次归一化误差。把“残差混合的非扩张”扩大成“训练必定稳定”，超出了数学结论的范围。[3, §2.2]

### 8.3 Muon：改变更新矩阵的几何形状

AdamW 主要通过逐元素的动量和二阶矩估计缩放更新。Muon 对矩阵参数的动量更新进行近似正交化。设动量矩阵：

{% raw %}
$$
M=U\Sigma V^T,
$$
{% endraw %}

其理想化的正交化方向为 {% raw %}<span class="math-inline">$UV^T$</span>{% endraw %}。直觉上，它弱化由不同奇异值幅度造成的更新方向失衡。实际不会为每块权重昂贵地做完整 SVD，而是通过 Newton–Schulz 型迭代近似。

V4 使用：

{% raw %}
$$
M_k=aM_{k-1}+b(M_{k-1}M_{k-1}^T)M_{k-1}
+c(M_{k-1}M_{k-1}^T)^2M_{k-1}.
$$
{% endraw %}

开始先按 Frobenius 范数归一化；前 8 次迭代用 {% raw %}<span class="math-inline">$(a,b,c)=(3.4445,-4.7750,2.0315)$</span>{% endraw %}，后 2 次改用 {% raw %}<span class="math-inline">$(2,-1.5,0.5)$</span>{% endraw %}。还配合 Nesterov 动量、weight decay 和更新 RMS 重缩放。

Embedding、预测头、RMSNorm 权重以及 mHC 的静态偏置和门控因子等仍使用 AdamW。**Muon 是优化器，不是一个 {% raw %}<span class="math-inline">$L_{\mathrm{Muon}}$</span>{% endraw %}；正交化的是更新方向，也不等于把模型权重永久约束为正交矩阵。** 其收益依赖参数类型和训练设置，不能简单写成“AdamW 总是收敛更慢”。[3, §2.4]

## 九、V4 的 MoE 与稳定性：值得单独展开的变化

### 9.1 亲和度改成平方根 softplus

学习路由层使用：

{% raw %}
$$
s_{i,t}=\sqrt{\operatorname{softplus}(z_{i,t})}
=\sqrt{\log(1+e^{z_{i,t}})}.
$$
{% endraw %}

与 sigmoid 相比，它仍保持正值，但正半轴不被限制在 1 以下。渐近地：

{% raw %}
$$
z\ll0:\ s(z)\approx e^{z/2},\qquad
z\gg0:\ s(z)\approx\sqrt z.
$$
{% endraw %}

这说明它保留了较大正 logit 之间的动态范围，同时增长受到平方根压缩。其导数为：

{% raw %}
$$
s'(z)=\frac{\sigma(z)}{2\sqrt{\operatorname{softplus}(z)}}.
$$
{% endraw %}

这是函数性质的解释，**不是独立消融证明该函数必然优于 sigmoid**。路由仍结合负载偏置与选中权重归一化，不能只替换激活函数就声称复现了 V4。

### 9.2 Hash 路由：没有学习型选择器，不等于专家不学习

前 3 层使用基于 token ID 的预定义 Hash 路由，替换之前的 dense FFN。可抽象写成：

{% raw %}
$$
\mathcal S_t=\mathcal H(x_t),\qquad |\mathcal S_t|=K_r.
$$
{% endraw %}

它把 token ID 映射到一组目标专家。不能用一个“hash 对专家数取模”的单专家公式，冒充实际 Top-6 的完整配置。

Hash 选择不根据当前上下文动态学习，但**被选中的专家网络仍通过梯度训练**。同时，token 频率并不均匀，所以“Hash 天然保证 token 负载均衡”不成立：即使词表条目分得均匀，热门 token 也可能集中产生负载。它节省了早层学习型路由的部分开销，并引入稳定分配，但也牺牲了该选择器的上下文适应性。[3, §2.1、§4.2]

### 9.3 不再限制目标节点数，但必须重做通信组织

移除节点数约束给路由更多自由，也可能让通信更难处理。V4 以更细粒度的专家并行和通信—计算重叠匹配这一变化。不能把它描述成没有代价的“去限制”。

### 9.4 Anticipatory Routing：打断异常值与路由的同步反馈

V4 报告观察到，loss spike 与 MoE 异常值及路由反馈有关。Anticipatory Routing 在当前步 {% raw %}<span class="math-inline">$t$</span>{% endraw %} 使用当前参数进行特征计算，却用历史参数预先算好的专家索引：

{% raw %}
$$
\text{features}:\theta_t,\qquad
\text{routing indices}:\theta_{t-\Delta t}.
$$
{% endraw %}

它只解耦这里的路由索引，不是把整步训练都退回旧模型。报告采用按需触发：检测 spike、短回滚、暂时启用，再恢复普通训练。启用期间存在额外前向计算开销；报告说明这种临时模式可控制总体成本。

这是一个经验稳定化方法，其完整理论机制仍未被报告完全解释。把它列出来，比笼统说“Muon 让 V4 更稳定”更接近真实训练过程。[3, §4.2.3]

### 9.5 SwiGLU clamping：在异常放大前截住数值

SwiGLU 将两个分支相乘，一个分支的异常值可能通过乘法放大输出。V4 将线性分支限制在 {% raw %}<span class="math-inline">$[-10,10]$</span>{% endraw %}，门分支的上界限制为 10。代价是裁剪区间外的信息与梯度受到影响，收益是抑制极端值带来的不稳定。

这仍属于前向计算与稳定性策略，不是新增一个名为“clamping loss”的损失。

## 十、Loss 总账：先分阶段，才能说“完整”

![预训练 Loss 对照](/images/deepseek-v2-v3-v4-moe-attention-loss/05_losses.png)

*图 8：这是主干预训练目标的对照。V4 的索引器训练有独立阶段；后训练的 SFT、GRPO 与 OPD 见下文，不能省略后就声称覆盖整个训练过程。*

### 10.1 V4 主干预训练：系数已明确，不需要写成未知的 ε

按 V4 报告继承的 MTP 与均衡设置，主干目标可概括为：

{% raw %}
$$
\boxed{L_{\mathrm{V4,backbone}}=L_{\mathrm{LM}}+
\lambda L_{\mathrm{MTP}}+10^{-4}L_{\mathrm{seq}}.}
$$
{% endraw %}

多数训练阶段 {% raw %}<span class="math-inline">$\lambda=0.3$</span>{% endraw %}，学习率开始衰减时调整为 0.1。不要把 V3 的“前 10T / 后 4.8T”时间表照搬到训练 32T / 33T token 的 V4。V4 的均衡偏置更新速度为 0.001，序列均衡 loss 系数明确为 0.0001。

因此，不能声称 V4 的序列均衡系数比 V3 更大；也不能叙述成“V3 删光辅助 loss，V4 又加回来”。[3, §4.2.2]

### 10.2 索引器训练：Top-K 自己不会告诉索引器该选谁

CSA 的离散 Top-K 选择不能被简单当作普通可微操作。V4 报告明确介绍了从 dense attention 到 indexer warmup，再到 sparse attention 的训练过程：Flash 先进行 dense warmup，并在更长序列阶段引入稀疏化；Pro 的 dense 阶段更长。

为了理解这种索引器如何获得监督，可以用 DSA 类方法的分布匹配思路表示：

{% raw %}
$$
L_{\mathrm{index}}^{\mathrm{illustrative}}
=\frac1{|\mathcal Q|}\sum_{t\in\mathcal Q}
D_{\mathrm{KL}}\!\left(\operatorname{sg}(A_t)\parallel
\operatorname{softmax}(I_{t,:})\right).
$$
{% endraw %}

这里 {% raw %}<span class="math-inline">$A_t$</span>{% endraw %} 是由主注意力构造的目标分布，{% raw %}<span class="math-inline">$I$</span>{% endraw %} 是索引分数，{% raw %}<span class="math-inline">$\operatorname{sg}$</span>{% endraw %} 表示停止目标侧梯度。其直觉是让便宜的索引器模仿昂贵注意力认为重要的位置。

**这只是解释索引监督的示意式，不冒充 V4 报告给出的完整训练实现。** V4 报告没有在对应设置段落完整列出索引器 loss 的全部系数、监督集合和梯度细节，因此本文不编造一个 {% raw %}<span class="math-inline">$\beta$</span>{% endraw %} 并将其硬塞进已确认的主干总式。要精确复现，仍需索引器训练代码或更详细配置；只看推理代码无法确定训练目标。[3, §4.2.2；5]

### 10.3 SFT：目标形式类似交叉熵，监督范围变了

给定问题 {% raw %}<span class="math-inline">$q$</span>{% endraw %} 与示范回答 {% raw %}<span class="math-inline">$y$</span>{% endraw %}，典型回答区间的 SFT 目标为：

{% raw %}
$$
L_{\mathrm{SFT}}=-\mathbb E_{(q,y)}
\left[\frac1{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid q,y_{\lt t})\right].
$$
{% endraw %}

它优化“在这个指令条件下模仿示范回答”，而预训练主要优化海量文本的续写。实际 loss mask 可能涵盖规定的 assistant 内容、推理或工具调用字段，要以对应训练格式为准。

高质量数学、代码与推理数据影响 SFT 能学到什么。V3 报告还描述了从推理模型产生的数据中蒸馏能力；使用教师生成文本进行 SFT，与后面直接匹配教师概率分布的 OPD，不是同一种监督粒度。

### 10.4 GRPO：从“模仿标准答案”转为“提高高奖励输出的概率”

V2、V3 使用 GRPO，V4 的领域专家训练也沿用这一类方法。对同一个问题，旧策略生成 {% raw %}<span class="math-inline">$G$</span>{% endraw %} 个回答，奖励为 {% raw %}<span class="math-inline">$r_1,\ldots,r_G$</span>{% endraw %}，组内优势可写为：

{% raw %}
$$
\widehat A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\epsilon_{\mathrm{num}}}.
$$
{% endraw %}

这里 {% raw %}<span class="math-inline">$\epsilon_{\mathrm{num}}$</span>{% endraw %} 是实现中的数值稳定项，不是 PPO 的裁剪阈值。若奖励为 {% raw %}<span class="math-inline">$(0,0,1,1)$</span>{% endraw %}，组内标准差按总体口径为 0.5，则优势约为 {% raw %}<span class="math-inline">$(-1,-1,+1,+1)$</span>{% endraw %}。

为清晰说明实际 token 级更新，写出一种常见 GRPO 形式：

{% raw %}
$$
r_{i,t}(\theta)=\frac{\pi_\theta(y_{i,t}\mid q,y_{i,\lt t})}
{\pi_{\mathrm{old}}(y_{i,t}\mid q,y_{i,\lt t})},
$$
{% endraw %}

{% raw %}
$$
J_{\mathrm{GRPO}}=\mathbb E\left[
\frac1G\sum_{i=1}^{G}\frac1{|y_i|}\sum_t
\left\{\min\!\left(r_{i,t}\widehat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\widehat A_i\right)
-\beta\widehat D_{\mathrm{KL},i,t}\right\}\right].
$$
{% endraw %}

训练最小化 {% raw %}<span class="math-inline">$L_{\mathrm{RL}}=-J_{\mathrm{GRPO}}$</span>{% endraw %}。报告可能用序列概率的紧凑记号表达目标；上式用于解释 token 级机制，不宣称所有版本的长度归一化、奖励组合和超参数完全相同。

逐项来看：

1. **优势 {% raw %}<span class="math-inline">$\widehat A_i$</span>{% endraw %}**：告诉模型这个答案比同题其他答案好还是差；组内相对值替代了一个独立 critic 的部分作用。
2. **新旧策略比值 {% raw %}<span class="math-inline">$r_{i,t}$</span>{% endraw %}**：衡量模型相对采样时策略改变了多少。
3. **clip 与 min**：限制一次更新过度放大某些答案的倾向，并不等于把所有梯度机械裁到同一范围。
4. **参考策略 KL**：约束模型不要过度偏离参考策略。{% raw %}<span class="math-inline">$\pi_{\mathrm{old}}$</span>{% endraw %} 用于采样与重要性比率；{% raw %}<span class="math-inline">$\pi_{\mathrm{ref}}$</span>{% endraw %} 用于偏离惩罚，角色不同。

常见的逐样本 KL 估计使用 {% raw %}<span class="math-inline">$a=\pi_{\mathrm{ref}}(y_t|h)/\pi_\theta(y_t|h)$</span>{% endraw %}，写为 {% raw %}<span class="math-inline">$a-\log a-1$</span>{% endraw %}。它与直接在整个词表求和的精确 KL，在计算和方差上不同。

数学、代码可以使用规则和测试奖励；开放写作等任务需要更复杂的评价。V4 的领域训练还按推理强度调整长度惩罚与上下文设置，并使用 rubric 引导的生成式奖励模型。报告没有完整披露每种奖励的全部系数，因而不能凭空写出一套统一的“最终 RL loss 数值配方”。[1, §4.2；2, §5.2；3, §5.1.1]

### 10.5 V4 的 OPD：让学生在自己会遇到的状态上向教师学习

V4 先分别训练多个领域教师，再通过多教师 on-policy distillation 把能力整合到一个学生。这里的“教师 / specialist”是完整模型，**不是 MoE 层里的小型 FFN 专家**。

![多教师 OPD](/images/deepseek-v2-v3-v4-moe-attention-loss/09_opd.png)

*图 9：学生生成自己的轨迹，相关领域教师在这些状态上提供完整词表分布，学生进行概率层面的对齐。*

V4 报告的目标为：

{% raw %}
$$
L_{\mathrm{OPD}}(\theta)=\sum_{i=1}^{N}w_i
D_{\mathrm{KL}}\left(\pi_\theta\parallel\pi_{E_i}\right).
$$
{% endraw %}

把它展开到学生轨迹中的一个上下文状态 {% raw %}<span class="math-inline">$h$</span>{% endraw %}：

{% raw %}
$$
D_{\mathrm{KL}}(p_\theta\parallel p_E)
=\sum_{v\in\mathcal V}p_\theta(v\mid h)
\log\frac{p_\theta(v\mid h)}{p_E(v\mid h)}.
$$
{% endraw %}

三个要点决定了它与普通“用老师答案训练学生”的区别。

**第一，轨迹来自学生。** 学生会走到与教师不同的中间状态。让教师在学生实际到达的状态上指导，可以减少只模仿教师轨迹导致的分布错配。

**第二，方向是反向 KL：学生相对教师。** 如果学生给教师认为很差的 token 分配了较大概率，该 token 会对 loss 产生较大贡献。这里不能把 {% raw %}<span class="math-inline">$D_{\mathrm{KL}}(\pi_\theta\parallel\pi_E)$</span>{% endraw %} 与反方向互换。

**第三，V4 使用完整词表的 logit 蒸馏。** 只取采样到的一个 token 估计概率差，成本较低但梯度估计方差较大；完整词表 KL 使用每个状态下更丰富的分布信息。V4 通过教师调度、隐藏状态传输和专门内核降低其工程代价。[3, §5.1.2、§5.2.2]

例如，学生对两个候选词给出 {% raw %}<span class="math-inline">$(0.6,0.4)$</span>{% endraw %}，教师给出 {% raw %}<span class="math-inline">$(0.9,0.1)$</span>{% endraw %}，该状态的 KL 为：

{% raw %}
$$
0.6\log\frac{0.6}{0.9}+0.4\log\frac{0.4}{0.1}\approx0.311.
$$
{% endraw %}

若学生变为 {% raw %}<span class="math-inline">$(0.85,0.15)$</span>{% endraw %}，该值降至约 0.012。损失不仅关心“最终采到哪个词”，还关心分布怎样偏离教师。

教师并非每个位置无差别地全部参与。领域选择和权重 {% raw %}<span class="math-inline">$w_i$</span>{% endraw %} 要匹配任务，例如数学问题主要向数学教师学习。报告使用十多个领域教师整合一个学生，这也是 V4 创新叙事里不能缺少的一环。

### 10.6 FP4 QAT：训练模型适应量化，不是再加一条损失

V4 在**后训练阶段**引入 MXFP4 QAT，覆盖 MoE 专家权重与 CSA indexer 的 QK 路径。因此不能说整个 V4 预训练从头到尾都是 FP4。[3, §5.2.1]

对专家权重，其机制可简化为：

{% raw %}
$$
\widetilde W=\operatorname{Dequant}_{\mathrm{FP8}}
\left(Q_{\mathrm{FP4}}(W_{\mathrm{master}})\right),\qquad
\frac{\partial\widetilde W}{\partial W_{\mathrm{master}}}\approx I.
$$
{% endraw %}

主权重以 FP32 保存，前向先体现 FP4 量化误差，再用相应 FP8 表示进入既有计算框架；反向用 straight-through estimator 把梯度传回主权重。这里的 FP4→FP8 可精确表示，建立在报告说明的格式与块尺度条件上，不代表原始 FP32→FP4 量化无损。

它改变的是损失函数所看到的前向模型 {% raw %}<span class="math-inline">$f_{\widetilde W}$</span>{% endraw %}，从而迫使参数适应部署时的低精度误差。原有 SFT、RL 或 OPD 目标仍可使用，不需要人为添加一个 {% raw %}<span class="math-inline">$L_{\mathrm{FP4}}$</span>{% endraw %}。

在推理和 RL rollout 时，直接使用原生 FP4 权重，以降低加载带宽并使采样行为与部署一致。报告还将 index score 从 FP32 降为 BF16，得到其设置下约 2 倍 Top-K 选择器加速与 99.7% 条目召回率；这不是整模型加速 2 倍，也不是任务准确率 99.7%。

### 10.7 一张表分清哪些东西会出现在 loss 里

| 项目 | 解决的问题 | 是否作为显式目标项 |
|---|---|---|
| LM / SFT 交叉熵 | 预测文本 / 模仿指令回答 | 是 |
| MTP 交叉熵 | 加密未来预测监督 | 是，主干预训练的加权辅助项 |
| 专家 / 设备 / 通信均衡 | 防路由集中与硬件拥堵 | V2 中是 |
| 极轻量序列均衡 | 防单序列极端不均 | V3、V4 中是 |
| 负载偏置更新 | 根据负载调节选择机会 | 否，无梯度反馈控制 |
| 索引器监督 | 学会检索注意力条目 | 需单列其训练阶段和配置 |
| GRPO | 提高高奖励行为概率 | RL 阶段目标 |
| OPD 反向 KL | 吸收领域教师分布 | V4 能力整合阶段目标 |
| MLA / CSA / HCA / mHC | 改变网络结构与状态传递 | 本身不是新增 loss |
| Muon / AdamW | 将梯度变成参数更新 | 优化器，不是任务 loss |
| FP8 / FP4 QAT / clamping | 数值效率与稳定性 | 训练计算策略，本身不必增加 loss |

## 十一、创新的边界：读完之后应该保留哪些判断？

**V2 的关键，是把“专家分工”和“历史状态压缩”同时落实。** 细粒度 + 共享专家提升条件计算的组织能力，MLA 降低 KV 缓存开销；三类均衡目标则让它在真实硬件上能被训练。首层 dense 是一种架构取舍，不能仅凭“还不够稀疏”就判定为缺点。

**V3 的关键，是把任务学习与资源分配更好地分开。** 路由偏置替代了强的主要均衡约束，轻量序列 loss 防极端情况；MTP 为主干提供额外预测监督；FP8 与 DualPipe 则支撑更大的训练规模。收益来自配合，不能全部归功于单一模块。

**V4 的关键，是将长上下文从容量指标变成可运转的系统。** CSA/HCA 减少历史读取与存储；mHC 和 Muon 改变表示传递与更新方式；训练稳定化处理真实的 spike；多教师 OPD 整合领域能力；FP4 QAT 降低部署成本。

这些设计仍然有明确代价：压缩可能丢信息，稀疏选择可能漏掉重要历史；多路残差与索引器增加实现复杂度；Hash 路由缺少上下文适应；低精度对内核与硬件提出要求。长窗口也不等于模型能可靠利用窗口中的每条信息。判断“创新是否有用”，最终仍需关注消融、统一口径的质量评估和真实服务吞吐，而不只是参数与压缩倍率。

## 附录：公式阅读速查

| 符号 | 含义 |
|---|---|
| {% raw %}<span class="math-inline">$T,d$</span>{% endraw %} | 序列长度、主干隐藏维度 |
| {% raw %}<span class="math-inline">$N_r,N_s,K_r$</span>{% endraw %} | 路由专家数、共享专家数、每 token 选中的路由专家数 |
| {% raw %}<span class="math-inline">$s_{i,t},b_i,g_{i,t}$</span>{% endraw %} | 原始亲和度、负载偏置、输出混合权重 |
| {% raw %}<span class="math-inline">$f_i,P_i$</span>{% endraw %} | 归一化硬选择频率、平均软概率 |
| {% raw %}<span class="math-inline">$D,M$</span>{% endraw %} | V2 专家设备总数、单 token 最多访问设备数 |
| {% raw %}<span class="math-inline">$d_c,d_h^R$</span>{% endraw %} | MLA 的 KV 压缩维度、位置 key 维度 |
| {% raw %}<span class="math-inline">$D_{\mathrm{MTP}},\lambda$</span>{% endraw %} | 额外预测深度、MTP 权重 |
| {% raw %}<span class="math-inline">$m,m',k,w$</span>{% endraw %} | CSA/HCA 压缩率、注意力 Top-K、局部窗口大小 |
| {% raw %}<span class="math-inline">$\pi_{\mathrm{old}},\pi_{\mathrm{ref}},\pi_E$</span>{% endraw %} | 采样旧策略、KL 参考策略、蒸馏教师 |
| {% raw %}<span class="math-inline">$\operatorname{sg}$</span>{% endraw %} | stop-gradient，停止梯度 |

## 参考资料与核验范围

1. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**，本文核对 v5 的架构、均衡公式、训练与对齐章节。<https://arxiv.org/html/2405.04434v5>
2. **DeepSeek-V3 Technical Report**，本文核对 v2 的路由、MTP、FP8、DualPipe、训练调度与 GRPO 章节。<https://arxiv.org/html/2412.19437v2>
3. **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence**，本文核对所访问 v1 的架构、模型设置、训练稳定性、OPD 与 QAT 章节。<https://arxiv.org/html/2606.19348v1>；官方预览发布说明：<https://api-docs.deepseek.com/news/news260424>；官方模型卡：<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro>。
4. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**，创新来源追溯。<https://arxiv.org/abs/2401.06066>
5. **DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention**，DSA 技术脉络与官方资料入口。<https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>

本文公式使用统一符号重写，明确标注了讲解性简化与归一化差异。未公开的训练细节保持为未确认，不用第三方猜测补齐。配图均为依据报告重新绘制的技术示意；示例中的人为数值不代表模型实测结果。
