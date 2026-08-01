---
type: mixed
density: balanced
style: blueprint
image_count: 4
language: zh-CN
watermark: false
---

## Illustration 1

**Position**: 开篇 / 一条样本的完整旅程
**Purpose**: 用一张图建立从 JSONL 到 total loss 的全局坐标系。
**Visual Content**: JSONL → prompt `[B,P]` → rollout `[B,P+R]`/`[B,R]` → reward/value → GAE → Actor/Critic losses → scalar。
**Type Application**: flowchart
**Filename**: 01-ppo-end-to-end.svg

## Illustration 2

**Position**: Actor、Critic、Reward、Reference 角色分工
**Purpose**: 消除“Critic 是 Reward Model”“old policy 是 Reference”等常见混淆。
**Visual Content**: 四角色卡片、输入输出 shape、更新/冻结状态和连接关系。
**Type Application**: framework
**Filename**: 02-actor-critic-roles.svg

## Illustration 3

**Position**: TD residual 与 GAE
**Purpose**: 直观看到终局 reward 怎样衰减传播给更早 token。
**Visual Content**: 三 token 例子、`token_rewards=[0,0,1]`、`old_values=[0.2,0.3,0.4]`、反向递推和结果。
**Type Application**: infographic
**Filename**: 03-gae-credit-assignment.svg

## Illustration 4

**Position**: Loss 设计总览
**Purpose**: 对比 Actor clip、Reference KL 与 Critic value clip 的输入、目标和约束对象。
**Visual Content**: 三栏比较，`ε=0.2`、`β=0.02`、`cV=0.5`，最后汇入 total loss。
**Type Application**: comparison
**Filename**: 04-ppo-losses.svg
