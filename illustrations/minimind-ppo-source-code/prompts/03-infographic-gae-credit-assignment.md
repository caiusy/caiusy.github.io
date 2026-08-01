---
illustration_id: 03
type: infographic
style: blueprint
---

GAE 如何把终局奖励传回更早 token - Data Visualization

Layout: horizontal three-token timeline with right-to-left curved arrows and a lower calculation zone.

ZONES:
- Top: token t0, t1, t2 with `token_rewards=[0,0,1]`
- Middle: old values `[0.2,0.3,0.4]`, terminal next value 0
- Bottom: `delta=[0.1,0.1,0.6]`, `raw advantages=[0.7365,0.6700,0.6000]`, `returns=[0.9365,0.9700,1.0000]`

LABELS: `delta_t = r_t + gamma V_(t+1) - V_t`, `A_t = delta_t + gamma lambda A_(t+1)`, `gamma=1.0`, `lambda=0.95`, 从后向前, 信用分配.
COLORS: navy #081526; terminal reward amber #FFBE55; TD delta cyan #42D3FF; advantage coral #FF6F7D; returns mint #58E0B5.
STYLE: data-first blueprint infographic, large legible numbers, thick backward arrows, no metaphorical imagery.
ASPECT: 16:9.
