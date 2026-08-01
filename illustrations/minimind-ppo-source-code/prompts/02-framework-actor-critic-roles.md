---
illustration_id: 02
type: framework
style: blueprint
---

PPO 四个模型角色 - Conceptual Framework

STRUCTURE: 2×2 card matrix around a central rollout trajectory.

NODES:
- Actor - selects next token, logits `[B,L,6400]`, trainable
- Critic - predicts state value, values `[B,R]`, trainable
- Reward Model - scores completed answer, reward `[B]`, frozen
- Reference Model - provides ref logps `[B,R]`, frozen

RELATIONSHIPS: Actor creates trajectory; Reward scores it; Critic estimates it; Reference constrains Actor via KL. Add a separate note: old policy is rollout-time old logps, not Reference.
LABELS: Actor 选 token, Critic 估未来, Reward 评整条, Reference 防跑偏, old logps ≠ Reference.
COLORS: navy #081526; Actor cyan #42D3FF; Critic mint #58E0B5; Reward amber #FFBE55; Reference violet #A88BFF.
STYLE: technical blueprint framework, uniform cards, large Chinese labels, precise arrows, generous whitespace.
ASPECT: 16:9.
