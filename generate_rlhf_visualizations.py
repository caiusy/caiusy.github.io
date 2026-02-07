"""
生成 InstructGPT/RLHF 详细可视化图表
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def create_sft_example_visualization():
    """创建 SFT 训练实例的详细可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SFT (Supervised Fine-Tuning) 详细示例', fontsize=16, fontweight='bold')
    
    # 1. 输入输出示例
    ax1 = axes[0, 0]
    ax1.axis('off')
    ax1.set_title('1. 训练数据格式', fontsize=14, fontweight='bold')
    
    prompt_text = "Prompt:\nExplain quantum computing\nto a 10-year-old"
    response_text = "Response:\nQuantum computers are like\nmagic computers that can try\nmany answers at once!"
    
    # Prompt box
    prompt_box = FancyBboxPatch((0.05, 0.55), 0.4, 0.35, 
                                boxstyle="round,pad=0.02", 
                                edgecolor='blue', facecolor='lightblue', 
                                linewidth=2, alpha=0.3)
    ax1.add_patch(prompt_box)
    ax1.text(0.25, 0.72, prompt_text, ha='center', va='center', 
             fontsize=11, weight='bold', color='darkblue')
    
    # Response box
    response_box = FancyBboxPatch((0.55, 0.55), 0.4, 0.35,
                                  boxstyle="round,pad=0.02",
                                  edgecolor='green', facecolor='lightgreen',
                                  linewidth=2, alpha=0.3)
    ax1.add_patch(response_box)
    ax1.text(0.75, 0.72, response_text, ha='center', va='center',
             fontsize=11, weight='bold', color='darkgreen')
    
    # Loss calculation
    loss_box = FancyBboxPatch((0.3, 0.1), 0.4, 0.3,
                              boxstyle="round,pad=0.02",
                              edgecolor='red', facecolor='lightyellow',
                              linewidth=2, alpha=0.3)
    ax1.add_patch(loss_box)
    ax1.text(0.5, 0.25, "Loss = Cross-Entropy\n(只计算 Response 部分)\n\nMask: [0,0,0,...,1,1,1]",
             ha='center', va='center', fontsize=11, weight='bold', color='darkred')
    
    # Arrows
    arrow1 = FancyArrowPatch((0.25, 0.53), (0.5, 0.42),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='blue')
    arrow2 = FancyArrowPatch((0.75, 0.53), (0.5, 0.42),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='green')
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. Token 级别的维度变化
    ax2 = axes[0, 1]
    ax2.axis('off')
    ax2.set_title('2. Token 级别维度变化', fontsize=14, fontweight='bold')
    
    stages = ['Input IDs\n[B, S]', 'Embedding\n[B, S, H]', 'Transformer\n[B, S, H]', 
              'LM Head\n[B, S, V]', 'Logits\n[B, S, V]']
    y_positions = [0.8, 0.6, 0.4, 0.2, 0.05]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
    
    for i, (stage, y, color) in enumerate(zip(stages, y_positions, colors)):
        box = FancyBboxPatch((0.15, y-0.08), 0.7, 0.12,
                            boxstyle="round,pad=0.01",
                            edgecolor='black', facecolor=color,
                            linewidth=2)
        ax2.add_patch(box)
        ax2.text(0.5, y-0.02, stage, ha='center', va='center',
                fontsize=12, weight='bold')
        
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((0.5, y-0.08), (0.5, y_positions[i+1]+0.04),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='black')
            ax2.add_patch(arrow)
    
    # 添加具体数值
    ax2.text(0.9, 0.78, 'B=4\nS=30', ha='left', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.9, 0.58, 'H=768', ha='left', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.text(0.9, 0.18, 'V=50257', ha='left', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. Loss Mask 可视化
    ax3 = axes[1, 0]
    ax3.set_title('3. Loss Mask 机制', fontsize=14, fontweight='bold')
    
    # 创建示例 mask
    seq_len = 30
    prompt_len = 10
    mask = np.zeros(seq_len)
    mask[prompt_len:] = 1.0
    
    colors_mask = ['lightcoral' if m == 0 else 'lightgreen' for m in mask]
    ax3.bar(range(seq_len), np.ones(seq_len), color=colors_mask, edgecolor='black')
    ax3.axvline(x=prompt_len-0.5, color='red', linestyle='--', linewidth=3, label='Prompt/Response 边界')
    ax3.set_xlabel('Token Position', fontsize=12)
    ax3.set_ylabel('Mask Value', fontsize=12)
    ax3.set_ylim(0, 1.2)
    ax3.legend(fontsize=10)
    ax3.text(5, 1.1, 'Prompt\n(不计算 Loss)', ha='center', fontsize=11, weight='bold', color='red')
    ax3.text(20, 1.1, 'Response\n(计算 Loss)', ha='center', fontsize=11, weight='bold', color='green')
    
    # 4. 损失函数计算流程
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.set_title('4. 损失计算步骤', fontsize=14, fontweight='bold')
    
    steps = [
        "Step 1: 获取 logits\nlogits = model(input_ids)\n[B, S, V]",
        "Step 2: 计算交叉熵\nloss_per_token = CrossEntropy(logits, labels)\n[B, S]",
        "Step 3: 应用 mask\nmasked_loss = loss * mask\n[B, S]",
        "Step 4: 平均\nfinal_loss = masked_loss.sum() / mask.sum()\nScalar"
    ]
    
    y_steps = [0.85, 0.6, 0.35, 0.1]
    for i, (step, y) in enumerate(zip(steps, y_steps)):
        box = FancyBboxPatch((0.05, y-0.08), 0.9, 0.18,
                            boxstyle="round,pad=0.01",
                            edgecolor='purple', facecolor='lavender',
                            linewidth=2, alpha=0.5)
        ax4.add_patch(box)
        ax4.text(0.5, y, step, ha='center', va='center',
                fontsize=10, family='monospace')
        
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((0.5, y-0.09), (0.5, y_steps[i+1]+0.09),
                                   arrowstyle='->', mutation_scale=15,
                                   linewidth=2, color='purple')
            ax4.add_patch(arrow)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('source/images/instructgpt/sft_detailed_example.png', dpi=300, bbox_inches='tight')
    print("✅ 已生成: sft_detailed_example.png")
    plt.close()


def create_rm_comparison_visualization():
    """创建 Reward Model 对比学习可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Reward Model (RM) 详细原理', fontsize=16, fontweight='bold')
    
    # 1. Pairwise 对比示例
    ax1 = axes[0, 0]
    ax1.axis('off')
    ax1.set_title('1. Pairwise 排序数据', fontsize=14, fontweight='bold')
    
    prompt_box = FancyBboxPatch((0.35, 0.85), 0.3, 0.1,
                               boxstyle="round,pad=0.01",
                               edgecolor='blue', facecolor='lightblue',
                               linewidth=2)
    ax1.add_patch(prompt_box)
    ax1.text(0.5, 0.9, "Prompt: Explain AI", ha='center', va='center',
            fontsize=11, weight='bold')
    
    # Winner response
    winner_box = FancyBboxPatch((0.05, 0.5), 0.4, 0.25,
                               boxstyle="round,pad=0.01",
                               edgecolor='green', facecolor='lightgreen',
                               linewidth=3)
    ax1.add_patch(winner_box)
    ax1.text(0.25, 0.7, "Response A (Winner) ✓", ha='center', fontsize=11,
            weight='bold', color='darkgreen')
    ax1.text(0.25, 0.6, "AI is the simulation of\nhuman intelligence by\nmachines.", 
            ha='center', fontsize=10)
    
    # Loser response
    loser_box = FancyBboxPatch((0.55, 0.5), 0.4, 0.25,
                              boxstyle="round,pad=0.01",
                              edgecolor='red', facecolor='lightcoral',
                              linewidth=3)
    ax1.add_patch(loser_box)
    ax1.text(0.75, 0.7, "Response B (Loser) ✗", ha='center', fontsize=11,
            weight='bold', color='darkred')
    ax1.text(0.75, 0.6, "AI is magic computer\nstuff that does things\nwith data.",
            ha='center', fontsize=10)
    
    # Arrows and scores
    arrow1 = FancyArrowPatch((0.25, 0.48), (0.25, 0.35),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='green')
    arrow2 = FancyArrowPatch((0.75, 0.48), (0.75, 0.35),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='red')
    ax1.add_patch(arrow1)
    ax1.add_patch(arrow2)
    
    ax1.text(0.25, 0.25, "r_w = 4.2", ha='center', fontsize=12,
            weight='bold', color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax1.text(0.75, 0.25, "r_l = 2.1", ha='center', fontsize=12,
            weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Loss calculation
    ax1.text(0.5, 0.08, "Loss = -log(σ(r_w - r_l)) = -log(σ(2.1)) = 0.114",
            ha='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. Bradley-Terry 概率曲线
    ax2 = axes[0, 1]
    ax2.set_title('2. Bradley-Terry 概率函数', fontsize=14, fontweight='bold')
    
    delta_r = np.linspace(-6, 6, 200)
    prob = 1 / (1 + np.exp(-delta_r))
    
    ax2.plot(delta_r, prob, linewidth=3, color='blue', label='P(A > B) = σ(r_A - r_B)')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 标注关键点
    ax2.plot([2.1], [1/(1+np.exp(-2.1))], 'ro', markersize=12, label='当前示例')
    ax2.annotate(f'r_w - r_l = 2.1\nP = {1/(1+np.exp(-2.1)):.3f}',
                xy=(2.1, 1/(1+np.exp(-2.1))), xytext=(3.5, 0.7),
                fontsize=10, weight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax2.set_xlabel('r_winner - r_loser', fontsize=12)
    ax2.set_ylabel('P(winner > loser)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    
    # 3. 梯度分析
    ax3 = axes[1, 0]
    ax3.set_title('3. 梯度直觉 (∂Loss/∂r_w)', fontsize=14, fontweight='bold')
    
    delta_r_grad = np.linspace(-6, 6, 200)
    gradient = 1/(1+np.exp(delta_r_grad)) - 1  # σ(r_w - r_l) - 1
    
    ax3.plot(delta_r_grad, gradient, linewidth=3, color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 填充区域
    ax3.fill_between(delta_r_grad, gradient, 0, where=(delta_r_grad < 0),
                     alpha=0.3, color='red', label='r_w < r_l (需要增大 r_w)')
    ax3.fill_between(delta_r_grad, gradient, 0, where=(delta_r_grad > 0),
                     alpha=0.3, color='blue', label='r_w > r_l (微调即可)')
    
    ax3.set_xlabel('r_winner - r_loser', fontsize=12)
    ax3.set_ylabel('梯度值', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 添加注释
    ax3.text(-3, -0.3, "区分度差\n梯度大", ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    ax3.text(3, -0.05, "区分度好\n梯度小", ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    # 4. Loss 曲线
    ax4 = axes[1, 1]
    ax4.set_title('4. Loss 曲线分析', fontsize=14, fontweight='bold')
    
    delta_r_loss = np.linspace(-6, 6, 200)
    loss = np.log(1 + np.exp(-delta_r_loss))
    
    ax4.plot(delta_r_loss, loss, linewidth=3, color='darkred', label='Loss = -log(σ(Δr))')
    ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 标注关键点
    key_points = [-2, 0, 2, 4]
    for point in key_points:
        loss_val = np.log(1 + np.exp(-point))
        ax4.plot([point], [loss_val], 'o', markersize=10)
        ax4.text(point, loss_val + 0.3, f'Δr={point}\nL={loss_val:.2f}',
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax4.set_xlabel('r_winner - r_loser', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)
    ax4.set_ylim(0, 7)
    
    plt.tight_layout()
    plt.savefig('source/images/instructgpt/rm_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 已生成: rm_detailed_analysis.png")
    plt.close()


def create_ppo_mechanism_visualization():
    """创建 PPO 机制详细可视化"""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('PPO (Proximal Policy Optimization) 完整机制', fontsize=16, fontweight='bold')
    
    # 1. 四个模型的交互
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_title('1. 四模型协同工作流程', fontsize=14, fontweight='bold')
    
    models = [
        ('Actor\n(πθ)', 0.15, 'lightblue', '生成回答'),
        ('Critic\n(Vφ)', 0.35, 'lightgreen', '估计价值'),
        ('Ref Model\n(πref)', 0.55, 'lightyellow', 'KL 约束'),
        ('RM\n(rψ)', 0.75, 'lightcoral', '打分')
    ]
    
    for name, x, color, func in models:
        circle = Circle((x, 0.7), 0.08, facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, 0.7, name, ha='center', va='center', fontsize=11, weight='bold')
        ax1.text(x, 0.5, func, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 添加数据流箭头
    ax1.annotate('', xy=(0.35, 0.65), xytext=(0.22, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax1.text(0.285, 0.62, 'logits', fontsize=9, ha='center')
    
    ax1.annotate('', xy=(0.55, 0.75), xytext=(0.22, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax1.text(0.385, 0.78, 'KL penalty', fontsize=9, ha='center')
    
    ax1.annotate('', xy=(0.75, 0.65), xytext=(0.22, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.text(0.485, 0.62, 'response', fontsize=9, ha='center')
    
    # Loss 计算
    loss_box = FancyBboxPatch((0.25, 0.15), 0.5, 0.2,
                             boxstyle="round,pad=0.02",
                             edgecolor='purple', facecolor='lavender',
                             linewidth=3)
    ax1.add_patch(loss_box)
    ax1.text(0.5, 0.25, "PPO Loss = -min(r_t·A_t, clip(r_t, 0.8, 1.2)·A_t)\n+ 0.5·MSE(V, returns)",
            ha='center', va='center', fontsize=11, weight='bold', family='monospace')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. KL Penalty 可视化
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title('2. KL Penalty 作用', fontsize=12, fontweight='bold')
    
    kl_values = np.linspace(0, 10, 100)
    beta = 0.1
    penalty = -beta * kl_values
    
    ax2.plot(kl_values, penalty, linewidth=3, color='red', label=f'Penalty = -β·KL (β={beta})')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(kl_values, penalty, 0, alpha=0.3, color='red')
    
    ax2.set_xlabel('KL Divergence', fontsize=11)
    ax2.set_ylabel('Penalty', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.text(5, -0.2, "偏离越多\n惩罚越大", ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 3. PPO Clip 机制
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title('3. PPO Clip 限制更新幅度', fontsize=12, fontweight='bold')
    
    ratios = np.linspace(0, 2.5, 200)
    epsilon = 0.2
    advantages_pos = np.ones_like(ratios) * 2  # A > 0
    advantages_neg = np.ones_like(ratios) * (-2)  # A < 0
    
    # 原始目标 (无clip)
    obj_no_clip_pos = ratios * advantages_pos
    obj_no_clip_neg = ratios * advantages_neg
    
    # Clip 后目标
    clipped_ratio_pos = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    obj_clip_pos = np.minimum(ratios * advantages_pos, clipped_ratio_pos * advantages_pos)
    
    clipped_ratio_neg = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    obj_clip_neg = np.maximum(ratios * advantages_neg, clipped_ratio_neg * advantages_neg)
    
    ax3.plot(ratios, obj_no_clip_pos, '--', linewidth=2, color='gray', label='无 Clip (A>0)', alpha=0.5)
    ax3.plot(ratios, obj_clip_pos, linewidth=3, color='green', label='PPO Clip (A>0)')
    ax3.axvline(x=1-epsilon, color='blue', linestyle='--', linewidth=1.5, label=f'Clip 边界')
    ax3.axvline(x=1+epsilon, color='blue', linestyle='--', linewidth=1.5)
    
    ax3.set_xlabel('Probability Ratio (π_new/π_old)', fontsize=11)
    ax3.set_ylabel('Objective', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 2.5)
    
    # 4. GAE 时序差分
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_title('4. GAE 优势函数估计', fontsize=12, fontweight='bold')
    
    timesteps = np.arange(10)
    rewards = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 0.9, 0.7, 0.4, 0.2])
    values = np.array([2.0, 2.1, 2.3, 2.6, 3.0, 3.2, 3.1, 2.8, 2.3, 1.8])
    
    gamma = 0.99
    lam = 0.95
    
    # 计算 TD error
    td_errors = rewards + gamma * np.append(values[1:], 0) - values
    
    ax4.bar(timesteps, rewards, alpha=0.5, label='Rewards', color='blue')
    ax4.plot(timesteps, values, 'ro-', linewidth=2, markersize=8, label='Values V(s)')
    ax4.plot(timesteps, td_errors, 'g^--', linewidth=2, markersize=6, label='TD Error δ')
    
    ax4.set_xlabel('Timestep', fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 完整训练循环
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.set_title('5. PPO 训练步骤详解', fontsize=14, fontweight='bold')
    
    steps = [
        "① Rollout\nActor 生成回答\n收集轨迹数据",
        "② 计算奖励\nR = r_RM - β·KL",
        "③ 估计优势\nA = GAE(R, V)",
        "④ 更新 Actor\nπ ← π + ∇PPO_loss",
        "⑤ 更新 Critic\nV ← V + ∇MSE_loss"
    ]
    
    x_positions = np.linspace(0.1, 0.9, len(steps))
    for i, (step, x) in enumerate(zip(steps, x_positions)):
        box = FancyBboxPatch((x-0.08, 0.3), 0.16, 0.5,
                            boxstyle="round,pad=0.02",
                            edgecolor='purple', facecolor='lavender',
                            linewidth=2, alpha=0.6)
        ax5.add_patch(box)
        ax5.text(x, 0.55, step, ha='center', va='center', fontsize=10, weight='bold')
        
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((x+0.08, 0.55), (x_positions[i+1]-0.08, 0.55),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='purple')
            ax5.add_patch(arrow)
    
    # 循环箭头
    arrow_loop = FancyArrowPatch((x_positions[-1], 0.25), (x_positions[0], 0.25),
                                arrowstyle='->', mutation_scale=20,
                                linewidth=3, color='red',
                                connectionstyle="arc3,rad=-.5")
    ax5.add_patch(arrow_loop)
    ax5.text(0.5, 0.05, "重复迭代直到收敛", ha='center', fontsize=12,
            weight='bold', color='red')
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    
    plt.savefig('source/images/instructgpt/ppo_complete_mechanism.png', dpi=300, bbox_inches='tight')
    print("✅ 已生成: ppo_complete_mechanism.png")
    plt.close()


def create_comparison_table_visualization():
    """创建三阶段对比表可视化"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    ax.set_title('InstructGPT 三阶段完整对比', fontsize=16, fontweight='bold', pad=20)
    
    # 表格数据
    stages = ['Stage 1: SFT', 'Stage 2: RM', 'Stage 3: PPO']
    aspects = ['目标', '输入', '输出', '损失函数', '数据量', '训练时间', '关键挑战']
    
    data = [
        ['学会回答格式', '学会判断好坏', '优化回答质量'],
        ['Prompt + Response', 'Prompt + (Win, Lose)', 'Prompt only'],
        ['预测下一个 token', '预测偏好分数', '生成完整回答'],
        ['CrossEntropy\n(masked)', '-log σ(r_w - r_l)', 'PPO Clip + KL'],
        ['13K 样本', '33K 对比对', '31K prompts'],
        ['几小时', '1-2 天', '数天'],
        ['数据质量要求高', '标注一致性', '训练不稳定']
    ]
    
    # 创建表格
    colors = [['lightblue']*3, ['lightgreen']*3, ['lightyellow']*3,
              ['lightcoral']*3, ['plum']*3, ['wheat']*3, ['lightpink']*3]
    
    table = ax.table(cellText=data, rowLabels=aspects, colLabels=stages,
                    cellLoc='center', loc='center',
                    cellColours=colors,
                    colWidths=[0.3, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('darkblue')
        table[(0, i)].set_text_props(weight='bold', color='white', size=13)
    
    # 设置行标签样式
    for i in range(len(aspects)):
        table[(i+1, -1)].set_facecolor('darkgray')
        table[(i+1, -1)].set_text_props(weight='bold', color='white', size=12)
    
    plt.savefig('source/images/instructgpt/three_stages_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 已生成: three_stages_comparison.png")
    plt.close()


def create_math_intuition_visualization():
    """创建数学公式直觉可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('核心数学公式的几何直觉', fontsize=16, fontweight='bold')
    
    # 1. Cross Entropy 的含义
    ax1 = axes[0, 0]
    ax1.set_title('1. Cross Entropy: 衡量分布差异', fontsize=13, fontweight='bold')
    
    # 真实分布 vs 预测分布
    categories = ['Token A', 'Token B', 'Token C', 'Token D']
    true_dist = np.array([0.7, 0.2, 0.08, 0.02])
    pred_dist1 = np.array([0.65, 0.25, 0.08, 0.02])  # 接近
    pred_dist2 = np.array([0.3, 0.4, 0.2, 0.1])  # 远离
    
    x = np.arange(len(categories))
    width = 0.25
    
    ax1.bar(x - width, true_dist, width, label='真实分布 P', alpha=0.8, color='green')
    ax1.bar(x, pred_dist1, width, label='预测分布 Q₁ (好)', alpha=0.8, color='blue')
    ax1.bar(x + width, pred_dist2, width, label='预测分布 Q₂ (差)', alpha=0.8, color='red')
    
    # 计算 CE
    ce1 = -np.sum(true_dist * np.log(pred_dist1 + 1e-10))
    ce2 = -np.sum(true_dist * np.log(pred_dist2 + 1e-10))
    
    ax1.text(0.5, 0.9, f'CE(P, Q₁) = {ce1:.3f}', transform=ax1.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    ax1.text(0.5, 0.82, f'CE(P, Q₂) = {ce2:.3f}', transform=ax1.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax1.set_ylabel('概率', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Sigmoid 函数的直觉
    ax2 = axes[0, 1]
    ax2.set_title('2. Sigmoid: 压缩到 (0,1)', fontsize=13, fontweight='bold')
    
    x_sigmoid = np.linspace(-10, 10, 200)
    y_sigmoid = 1 / (1 + np.exp(-x_sigmoid))
    
    ax2.plot(x_sigmoid, y_sigmoid, linewidth=3, color='purple')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 标注关键点
    key_points = [(-5, 'Δr=-5\nP≈0'), (0, 'Δr=0\nP=0.5'), (5, 'Δr=5\nP≈1')]
    for x_val, text in key_points:
        y_val = 1 / (1 + np.exp(-x_val))
        ax2.plot(x_val, y_val, 'ro', markersize=10)
        ax2.text(x_val, y_val + 0.15, text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    ax2.set_xlabel('r_winner - r_loser', fontsize=11)
    ax2.set_ylabel('P(winner > loser)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # 3. KL 散度的几何意义
    ax3 = axes[1, 0]
    ax3.set_title('3. KL 散度: 两分布的"距离"', fontsize=13, fontweight='bold')
    
    # 两个高斯分布
    x_kl = np.linspace(-5, 5, 200)
    p = np.exp(-x_kl**2 / 2) / np.sqrt(2 * np.pi)
    q1 = np.exp(-(x_kl-0.5)**2 / 2) / np.sqrt(2 * np.pi)  # 接近
    q2 = np.exp(-(x_kl-2)**2 / 2.5) / np.sqrt(2 * np.pi * 2.5)  # 远离
    
    ax3.plot(x_kl, p, linewidth=3, label='参考分布 π_ref', color='green')
    ax3.plot(x_kl, q1, linewidth=3, label='策略 π₁ (KL小)', color='blue', linestyle='--')
    ax3.plot(x_kl, q2, linewidth=3, label='策略 π₂ (KL大)', color='red', linestyle='--')
    ax3.fill_between(x_kl, p, alpha=0.2, color='green')
    
    # 计算 KL
    kl1 = np.sum(p * np.log((p + 1e-10) / (q1 + 1e-10))) * (x_kl[1] - x_kl[0])
    kl2 = np.sum(p * np.log((p + 1e-10) / (q2 + 1e-10))) * (x_kl[1] - x_kl[0])
    
    ax3.text(0.5, 0.9, f'KL(π_ref || π₁) = {kl1:.3f}', transform=ax3.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    ax3.text(0.5, 0.82, f'KL(π_ref || π₂) = {kl2:.3f}', transform=ax3.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax3.set_xlabel('动作空间', fontsize=11)
    ax3.set_ylabel('概率密度', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. PPO Clip 的数学意义
    ax4 = axes[1, 1]
    ax4.set_title('4. PPO Clip: 限制策略更新', fontsize=13, fontweight='bold')
    
    r_values = np.linspace(0, 3, 200)
    epsilon = 0.2
    
    # Advantage > 0 的情况
    unclipped = r_values
    clipped = np.minimum(r_values, 1 + epsilon)
    
    ax4.plot(r_values, unclipped, '--', linewidth=2, color='gray', 
            label='无限制更新', alpha=0.5)
    ax4.plot(r_values, clipped, linewidth=3, color='blue', 
            label='PPO Clip 限制')
    ax4.axvline(x=1+epsilon, color='red', linestyle='--', linewidth=2,
               label=f'Clip 边界 (1+ε={1+epsilon})')
    ax4.axhline(y=1+epsilon, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # 填充受限区域
    ax4.fill_between(r_values, unclipped, clipped, where=(r_values > 1+epsilon),
                    alpha=0.3, color='red', label='被裁剪区域')
    
    ax4.set_xlabel('概率比 π_new / π_old', fontsize=11)
    ax4.set_ylabel('实际使用的比值', fontsize=11)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 3)
    ax4.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig('source/images/instructgpt/math_intuition.png', dpi=300, bbox_inches='tight')
    print("✅ 已生成: math_intuition.png")
    plt.close()


# 主函数
if __name__ == "__main__":
    print("🎨 开始生成 InstructGPT/RLHF 详细可视化图表...")
    print()
    
    create_sft_example_visualization()
    create_rm_comparison_visualization()
    create_ppo_mechanism_visualization()
    create_comparison_table_visualization()
    create_math_intuition_visualization()
    
    print()
    print("🎉 所有图表生成完成！")
    print("📁 保存位置: source/images/instructgpt/")
