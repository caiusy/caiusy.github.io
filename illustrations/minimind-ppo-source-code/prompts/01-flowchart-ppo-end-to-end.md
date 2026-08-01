---
illustration_id: 01
type: flowchart
style: blueprint
---

MiniMind PPO 从 JSONL 到 Total Loss - Process Flow

Layout: left-to-right main flow with one lower branch for value estimation, 16:9.

STEPS:
1. JSONL conversations - remove final empty assistant with conversations[:-1]
2. Chat template + tokenizer - prompt `[B,P]`, real sample `[1,561]`
3. Actor rollout - output `[1,667]`, completion `[1,106]`, old logps `[1,106]`
4. Reward + Critic - terminal reward `[1]`, old values `[1,106]`
5. TD + GAE - advantages and returns `[1,106]`
6. PPO update - Actor clip, Reference KL, Critic value clip
7. Total loss - scalar `[]`

CONNECTIONS: bold arrows; distinguish no-gradient rollout data from gradient-bearing current Actor/Critic paths.
LABELS: JSONL, prompt, rollout, reward, old values, GAE, returns, total loss, `[1,561]`, `[1,106]`, `[]`.
COLORS: navy background #081526; cyan #42D3FF for data, amber #FFBE55 for reward, mint #58E0B5 for value, coral #FF6F7D for loss.
STYLE: precise blueprint grid, crisp vector lines, rounded technical modules, large Chinese labels, no decorative people.
ASPECT: 16:9.
