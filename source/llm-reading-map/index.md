---
title: Multimodal LLM Reading Map
date: 2026-05-16
updated: 2026-05-16
type: topic
layout: page
description: 从 Transformer 出发的多模态大模型学习路线 — Transformer / BERT / GPT / RLHF / CLIP / BLIP / LLaVA / MoE / LoRA / QLoRA / RAG / Agent。
---

## Multimodal LLM Reading Map

不是阅读清单，而是一张**有顺序、有依赖、有复习问题**的学习地图。

```
Transformer
    ↓
   BERT ── GPT
              ↓
       InstructGPT / RLHF
              ↓
            CLIP
              ↓
            BLIP
              ↓
           LLaVA
              ↓
            MoE
              ↓
            LoRA
              ↓
           QLoRA
              ↓
             RAG
              ↓
            Agent
```

---

## 1. Transformer

- **核心问题**：如何让序列建模摆脱 RNN 的串行依赖？
- **前置知识**：注意力机制、位置编码、softmax、矩阵求导
- **读完能理解**：Self-Attention 为什么 work、QKV 三件套的几何意义、Layer Norm 放哪里
- **推荐文章**：TODO（"Attention Is All You Need" 精读）
- **复习问题**
  1. 为什么 attention 要除以 √d_k？
  2. multi-head 的本质是什么？
  3. positional encoding 为什么用三角函数？
  4. encoder-only / decoder-only / encoder-decoder 各适用什么任务？

---

## 2. BERT

- **核心问题**：怎么用大规模无标注语料学到通用语义表示？
- **前置知识**：Transformer encoder
- **读完能理解**：MLM 与 NSP 的设计动机、双向编码 vs 单向、fine-tune 范式
- **推荐文章**：TODO
- **复习问题**
  1. MLM 任务为什么用 [MASK] 而不是删除？
  2. NSP 任务后来为什么被否定？
  3. CLS token 的语义到底是什么？

---

## 3. GPT 系列

- **核心问题**：能否仅靠语言建模一个目标，学到所有语言能力？
- **前置知识**：Transformer decoder、autoregressive 训练
- **读完能理解**：GPT-1/2/3 的尺度律、in-context learning 的涌现
- **推荐文章**：TODO
- **复习问题**
  1. GPT 为什么选 decoder-only？
  2. in-context learning 的本质是什么？
  3. scaling law 揭示了什么？

---

## 4. InstructGPT / RLHF

- **核心问题**：怎么让大模型“听话”？
- **前置知识**：GPT、强化学习基础（PPO）、reward model 训练
- **读完能理解**：SFT → RM → PPO 三阶段、为什么 RLHF 比纯 SFT 强
- **推荐文章**：TODO
- **复习问题**
  1. 为什么需要 reward model？
  2. PPO 在 RLHF 里的具体角色？
  3. 为什么 RLHF 后模型变得更"礼貌"？

---

## 5. CLIP

- **核心问题**：能否让图像和文本共享同一语义空间？
- **前置知识**：Transformer、对比学习、ViT
- **读完能理解**：图文对比学习目标、zero-shot 分类的底层原理、temperature 的作用
- **推荐文章**：TODO（CLIP 论文精读）
- **复习问题**
  1. 为什么 CLIP 用对比学习而不是 caption 生成？
  2. temperature 起什么作用？
  3. zero-shot 分类背后的几何直觉？

---

## 6. BLIP

- **核心问题**：图文理解和图文生成怎么统一？
- **前置知识**：CLIP、Encoder-Decoder
- **读完能理解**：ITC / ITM / LM 三个目标、BLIP-2 的 Q-Former
- **推荐文章**：TODO
- **复习问题**
  1. ITC vs ITM 的区别？
  2. Q-Former 解决了什么问题？

---

## 7. LLaVA

- **核心问题**：怎么把现成的 LLM 变成"能看图的 LLM"？
- **前置知识**：CLIP、Vicuna/LLaMA、Instruction Tuning
- **读完能理解**：视觉编码器 + 投影层 + LLM 的对齐范式、两阶段训练（projection → end-to-end）
- **推荐文章**：[LLaVA 终极指南](/) (现有文章)、TODO 重写为《LLaVA 原理拆解：视觉编码器、投影层与语言模型如何对齐》
- **复习问题**
  1. 为什么投影层只用一个 MLP 就够了？
  2. 第一阶段为什么冻结视觉编码器和 LLM？
  3. LLaVA 和 BLIP-2 在对齐策略上有什么区别？

---

## 8. MoE (Mixture of Experts)

- **核心问题**：模型参数量爆炸时，能否让每次推理只激活一部分？
- **前置知识**：Transformer FFN、Router
- **读完能理解**：稀疏激活、负载均衡 loss、Top-K routing
- **推荐文章**：TODO
- **复习问题**
  1. MoE 的稀疏性体现在哪？
  2. router 是怎么训练的？
  3. 负载不均衡为什么是个大问题？

---

## 9. LoRA

- **核心问题**：能不能在不动原模型权重的情况下做高效微调？
- **前置知识**：矩阵低秩分解、Transformer 权重结构
- **读完能理解**：W + ΔW = W + BA 的低秩拆解、rank 的取舍、合并推理
- **推荐文章**：TODO
- **复习问题**
  1. LoRA 的 rank 越大越好吗？
  2. LoRA 加在哪些层效果最好？为什么？
  3. 推理时怎么把 LoRA 合并回原权重？

---

## 10. QLoRA

- **核心问题**：能在单卡 24G 上微调 65B 模型吗？
- **前置知识**：LoRA、量化（INT8 / NF4）、PagedAttention
- **读完能理解**：4-bit NF4 + double quantization + paged optimizer 三件套
- **推荐文章**：TODO
- **复习问题**
  1. NF4 vs INT4 的差异？
  2. double quantization 在量化什么？
  3. 显存到底省在哪一步？

---

## 11. RAG

- **核心问题**：模型没见过的知识怎么用？
- **前置知识**：CLIP/text embedding、向量数据库、Prompt Engineering
- **读完能理解**：retrieve → rerank → augment → generate 流程、chunk size 选择
- **推荐文章**：TODO
- **复习问题**
  1. chunk 太大太小各有什么问题？
  2. rerank 为什么有时是必要的？
  3. RAG 和 long context 是替代关系吗？

---

## 12. Agent

- **核心问题**：怎么让 LLM 不只生成文字，还能"做事"？
- **前置知识**：function calling、ReAct、Tool Use
- **读完能理解**：Plan-Execute、ReAct、Reflection 三种范式
- **推荐文章**：TODO
- **复习问题**
  1. function calling 和 ReAct 是同一回事吗？
  2. Agent 的失败模式有哪些？
  3. 多步 Agent 怎么避免"绕圈"？

---

## 怎么用这张地图

- **第一遍**：按顺序通读，每个节点至少能回答 1～2 个复习问题
- **第二遍**：选 3 个最感兴趣的节点深入读论文 + 复现关键代码
- **长期**：每写一篇相关文章，回到对应节点挂链接，让地图越来越实
