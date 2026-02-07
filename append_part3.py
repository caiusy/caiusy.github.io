
part3 = r'''
---

## 5. Stage 3: PPO (Proximal Policy Optimization) —— 自我进化

这是 RLHF 最复杂、最令人着迷，也是最难调优的阶段。

### 5.1 为什么要用强化学习？

有了 RM（裁判），我们理论上可以直接让模型生成 100 个回答，选 RM 分数最高的那个（Best-of-N 采样）。这在推理时很有用，但并没有**改变模型本身的参数**。

我们要的是**修改模型的权重**，让它下次直接生成高分回答。这就变成了 RL 问题：
- **Agent**: 语言模型 (Actor)
- **Environment**: 用户 Prompt
- **Action**: 生成下一个 Token
- **State**: 当前生成的文本
- **Reward**: RM 给出的分数

### 5.2 宏观架构：PPO 四大天王

PPO 训练过程涉及 4 个深度神经网络。这使得显存消耗巨大（通常需要 SFT 阶段的 3-4 倍）。

![PPO 完整机制](/images/instructgpt/ppo_complete_mechanism.png)

| 模型角色 | 符号 | 状态 | 作用 |
|:---|:---|:---:|:---|
| **Actor (策略模型)** | $\pi_{\theta}$ | **训练中** | 负责生成回答，我们的最终产品 |
| **Critic (价值模型)** | $V_{\phi}$ | **训练中** | 估计当前状态的未来价值，辅助 Actor 更新 |
| **Reward Model (裁判)** | $r_{\psi}$ | **冻结** | 给回答打分，提供原始奖励信号 |
| **Reference Model (参考)** | $\pi_{ref}$ | **冻结** | SFT 模型的副本，防止 Actor 跑偏 |

### 5.3 核心公式解析 1：KL Penalty (Beta)

**问题**：如果只最大化 Reward，Actor 会找到 RM 的漏洞（Reward Hacking），生成类似 "Very good good good..." 这样能骗过 RM 但人类无法阅读的乱码。

**解决方案**：我们要约束 Actor，让它的分布不要偏离 SFT 模型太远。

$$ R_{total}(x, y) = r_{RM}(x, y) - \beta \cdot \log \left( \frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)} \right) $$

- $\beta$ (KL 系数)：控制约束力度。通常取 0.02 ~ 0.1。
- **直觉**：如果 Actor 生成了 Ref 模型认为概率极低的词，这一项会变得非常大（惩罚重）。

### 5.4 核心公式解析 2：PPO Clip

Policy Gradient 最大的问题是更新步长难以控制。步长太大，策略崩溃；步长太小，训练太慢。

PPO 的核心创新在于 Clip 机制：

$$ \mathcal{L}^{CLIP} = \mathbb{E} \left[ \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t) \right] $$

其中 $r_t = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$ 是概率比率。

**直觉解释**：
- 如果当前策略比旧策略好太多（$r_t$ 很大），我们**截断**奖励，不让它更新太猛。
- 这就像给赛车加了限速器，保证在安全范围内加速。

### 5.5 核心公式解析 3：GAE (Generalized Advantage Estimation)

Critic 模型预测的是 $V(s)$（当前状态值多少分）。我们需要计算**优势函数 (Advantage)** $A(s, a)$：

> "这个动作比平均水平好多少？"

GAE 平衡了偏差（Bias）和方差（Variance）：

$$ A_t^{GAE} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + ... $$
$$ \text{其中 } \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$

### 5.6 完整 PPO 训练代码实现

```python
class PPOTrainer:
    def __init__(self, actor, critic, ref_model, reward_model, config):
        self.actor = actor
        self.critic = critic
        self.ref = ref_model
        self.rm = reward_model
        self.config = config

    def train_step(self, prompts):
        # 1. Rollout (采样)
        with torch.no_grad():
            # Actor 生成回答
            seq = self.actor.generate(prompts)
            
            # Ref 模型计算概率 (用于 KL)
            ref_logits = self.ref(seq).logits
            ref_logprobs = log_probs_from_logits(ref_logits, seq)
            
            # RM 打分
            rewards_raw = self.rm(seq)
            
        # 2. 计算 PPO 奖励 (Reward + KL Penalty)
        # 再次计算 Actor 的概率
        logits = self.actor(seq).logits
        logprobs = log_probs_from_logits(logits, seq)
        
        # KL = log(pi) - log(ref)
        kl = logprobs - ref_logprobs
        non_score_reward = -self.config.beta * kl
        rewards = non_score_reward.clone()
        rewards[:, -1] += rewards_raw  # 只在最后一步加上 RM 分数

        # 3. 计算 Advantage (GAE)
        values = self.critic(seq)
        advantages, returns = compute_gae(rewards, values)

        # 4. Update (PPO Loss)
        # 重新前向传播 (因为要计算梯度)
        new_logits = self.actor(seq).logits
        new_logprobs = log_probs_from_logits(new_logits, seq)
        
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # Clip Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss
        new_values = self.critic(seq)
        value_loss = F.mse_loss(new_values, returns)
        
        loss = policy_loss + 0.5 * value_loss
        loss.backward()
```

---

## 6. 实战总结与常见陷阱

### 6.1 常见陷阱 (Pitfalls)

1.  **Reward Hacking**：
    - **现象**：RM 分数很高，但生成的文本狗屁不通。
    - **原因**：RM 过拟合，或者 KL 约束太弱。
    - **解法**：增大 Beta，或者让 RM 见识更多负样本。

2.  **模式坍塌 (Mode Collapse)**：
    - **现象**：模型对所有问题都回答一样的套话。
    - **原因**：PPO 步长太大，或者 Entropy 正则化不够。
    - **解法**：调小 Learning Rate，增加 Entropy Loss。

3.  **Token 维度灾难**：
    - **现象**：OOM (Out of Memory)。
    - **解法**：使用 LoRA 微调 Actor/Critic，冻结大部分参数；或者使用 ZeRO-3 Offload。

### 6.2 RLHF vs DPO

2023 年提出的 **DPO (Direct Preference Optimization)** 正在挑战 PPO 的地位。

- **PPO**：需要训练 RM，需要采样，不稳定，显存占用大。
- **DPO**：**不需要 RM**，直接在 Preference 数据上优化 Policy。数学上证明了 DPO 等价于 RLHF 的最优解。

> **结论**：如果你是从头开始，建议先试 DPO。但理解 PPO 对深入掌握 LLM 对齐原理依然至关重要。

---

## 7. 结语：通往 AGI 的必经之路

InstructGPT 和 RLHF 的成功告诉我们：**让 AI 变大只是第一步，让 AI 变"好"才是关键。**

这不仅仅是技术问题，更是哲学问题：我们到底希望 AI 拥有什么样的价值观？

通过 SFT 教会它语言，通过 RM 告诉它对错，通过 PPO 让它自我完善——这像极了人类教育下一代的过程。而这，或许就是通往 AGI 的必经之路。

---

### 📚 参考文献与资源

1.  **InstructGPT**: [Training language models to follow instructions with human feedback (NeurIPS 2022)](https://arxiv.org/abs/2203.02155)
2.  **PPO**: [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347)
3.  **DeepSpeed-Chat**: Microsoft 的开源 RLHF 训练框架
4.  **Trlx**: CarperAI 开源的分布式 RLHF 库
'''

with open("source/_posts/InstructGPT-RLHF-Complete-Guide.md", "a") as f:
    f.write(part3)
