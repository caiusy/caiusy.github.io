
---

## 4. Stage 2: Reward Model (RM) —— 训练“裁判”

如果说 SFT 是教模型“如何回答”，那么 RM 就是教模型“分辨好坏”。

### 4.1 核心挑战：如何量化“好”？

让机器直接生成好答案很难，但让机器（或人）判断哪个答案更好相对容易。

**为什么选择 Ranking（排序）而不是 Rating（打分）？**

InstructGPT 论文中做了一个重要实验，对比了两种标注方式：
1.  **Rating**：直接给回答打分（1-5分）。
2.  **Ranking**：给出两个回答 A 和 B，让人选哪个更好。

**结果令人惊讶**：
- **Rating 一致性**：0.42 (Cohen's κ) —— 不同人打分差异巨大，标准很难统一。
- **Ranking 一致性**：**0.73** (Cohen's κ) —— 人类在“比较”时意见高度一致。

因此，RM 的训练数据是 **(Prompt, Winner, Loser)** 三元组。

### 4.2 Reward Model 架构

![Reward Model 详细分析](/images/instructgpt/rm_detailed_analysis.png)

- **基座模型**：通常使用 SFT 后的模型（去掉最后的 Unembedding 层）。
- **输出层**：将 hidden state 映射为 **Scalar（标量）**，即 `output_dim=1`。
- **输入**：`[CLS] Prompt [SEP] Response`
- **输出**：一个实数 $r$，代表该 Response 的质量得分。

### 4.3 数学推导：Bradley-Terry 模型

这是 RM 的灵魂。我们假设每个回答都有一个潜在的“质量得分” $r$。

根据 **Bradley-Terry 模型**，人类认为回答 $y_w$ (Winner) 比 $y_l$ (Loser) 好的概率为：

$$ P(y_w \succ y_l | x) = \frac{e^{r(x, y_w)}}{e^{r(x, y_w)} + e^{r(x, y_l)}} = \sigma(r(x, y_w) - r(x, y_l)) $$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 是 Sigmoid 函数。

**直觉解释**：
- 如果 $r_w$ 比 $r_l$ 大很多，$\sigma(r_w - r_l) \to 1$，概率接近 100%。
- 如果 $r_w \approx r_l$，$\sigma \approx 0.5$，很难区分。

### 4.4 损失函数推导 (Ranking Loss)

我们要最大化标注数据的似然概率。即最小化负对数似然（Negative Log Likelihood）：

$$ \mathcal{L}_{RM} = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r(x, y_w) - r(x, y_l)) \right] $$

展开 log-sigmoid：
$$ \mathcal{L}_{RM} = - \log \left( \frac{1}{1 + e^{-(r_w - r_l)}} \right) = \log(1 + e^{-(r_w - r_l)}) $$

### 4.5 梯度分析：为什么这个 Loss 有效？

让我们对 $r_w$ 求导，看看梯度是如何指导模型优化的：

$$ \frac{\partial \mathcal{L}}{\partial r_w} = \frac{1}{1 + e^{-(r_w - r_l)}} \cdot e^{-(r_w - r_l)} \cdot (-1) = \sigma(r_w - r_l) - 1 $$

同理：
$$ \frac{\partial \mathcal{L}}{\partial r_l} = 1 - \sigma(r_w - r_l) $$

**梯度直觉分析**：
1.  **当模型预测错误 ($r_w < r_l$)**：
    - $\sigma(r_w - r_l) \approx 0$
    - 梯度 $\approx -1$（很大！）
    - **效果**：模型受到强烈惩罚，拼命拉高 $r_w$，压低 $r_l$。

2.  **当模型预测正确且区分度很大 ($r_w \gg r_l$)**：
    - $\sigma(r_w - r_l) \approx 1$
    - 梯度 $\approx 0$（很小）
    - **效果**：模型“躺平”，不再浪费精力优化已经学好的样本。

这种机制让 RM 能够专注于那些**难以区分**的样本（Hard Negatives），效率极高。

### 4.6 生产级 RM 代码实现

```python
import torch
import torch.nn as nn

class GPTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 使用预训练的 Transformer 作为 backbone
        self.transformer = AutoModel.from_config(config)
        # 2. 这里的 config.hidden_size 通常是 768 (GPT2-Base) 或更大
        # 3. 输出维度是 1 (Scalar Score)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        # Transformer 前向传播
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state # [B, L, H]
        
        # 获取 EOS token 对应的 hidden state 作为句子表示
        # 假设 input_ids 已经被 padding 使得 EOS 在不同位置
        # 这里简化处理，取最后一个非 padding token
        rewards = self.v_head(hidden_states).squeeze(-1) # [B, L]
        
        return rewards

def compute_ranking_loss(chosen_rewards, rejected_rewards):
    """
    chosen_rewards: [B]  (Winner 的分数)
    rejected_rewards: [B] (Loser 的分数)
    """
    # 计算差值
    diff = chosen_rewards - rejected_rewards
    
    # Sigmoid + Log + Negative
    # PyTorch 的 LogSigmoid 数值稳定性更好
    loss = -torch.nn.functional.logsigmoid(diff).mean()
    
    return loss
```

### 4.7 数据增强技巧：K-pair Loss

为了提高数据利用率，InstructGPT 在每个 prompt 下采集 $K$ 个回答（例如 $K=9$），然后让标注员进行排序。

从 $K$ 个回答中，我们可以构建 $C_K^2 = \frac{K(K-1)}{2}$ 对比较数据。
例如 $K=9$ 时，一个 Prompt 就能产生 **36 对**数据！这大大降低了标注成本。

公式扩展为：
$$ \mathcal{L} = \frac{1}{C_K^2} \sum_{i=1}^K \sum_{j=i+1}^K \log \sigma(r(y_i) - r(y_j)) $$

---
