---
illustration_id: 04
type: comparison
style: blueprint
---

PPO 三类核心 Loss - Comparison View

Layout: three equal vertical panels feeding a bottom total-loss bar.

LEFT - Actor Clipped Loss:
- inputs: advantages, current logps, old logps `[B,R]`
- ratio clip `[0.8,1.2]`
- purpose: improve good tokens, suppress bad tokens, limit policy jump

CENTER - Reference KL:
- inputs: current logps, ref logps `[B,R]`
- coefficient `beta=0.02`
- purpose: keep language behavior near frozen reference

RIGHT - Critic Value Loss:
- inputs: current values, old values, returns `[B,R]`
- value clip `±0.2`, coefficient `cV=0.5`
- purpose: fit returns without value jump

BOTTOM: `L_total = L_actor + 0.02 L_KL + 0.5 L_value`, scalar `[]`; MoE aux optional and disabled in this run.
LABELS: Actor clip, Reference KL, Critic value clip, `[B,R] → []`, response mask.
COLORS: cyan #42D3FF, violet #A88BFF, mint #58E0B5, coral #FF6F7D, navy #081526.
STYLE: crisp blueprint comparison infographic, strong alignment, concise Chinese labels, generous spacing.
ASPECT: 16:9.
