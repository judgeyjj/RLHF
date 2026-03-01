# ===========================================================================
# ppo.py - Proximal Policy Optimization（近端策略优化）
# ===========================================================================
#
# 【面试核心理解】
# PPO 是所有 LLM RL 对齐算法的"祖师爷"。ChatGPT 的 RLHF 就用的它。
# 它最经典，但也最复杂、最吃显存。
#
# 核心公式：
#   Loss = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
#
# 角色分配（4 个模型）：
#   Actor (策略模型):   负责生成回答（做题的学生）
#   Critic (价值模型):  预测"这道题平时能拿多少分"（预测官）
#   Reference (参考):   冻住的原始模型（KL 散度的锚点）
#   Reward (奖励):      给回答打分（我们用规则代替）
#
# KL 散度在 PPO-RLHF 中的用法（InstructGPT 方式）:
#   不是作为额外的 Loss 项，而是融入到每个 token 的 reward 中:
#   adjusted_reward[t] = reward[t] - beta * (log_pi_new[t] - log_pi_ref[t])
#   然后用 adjusted_reward 去计算 GAE Advantage
#
# 缺点:
#   1. 需要 4 个模型同时在显存中 → 极其吃显存
#   2. 训练链条太长 (生成→打分→估值→GAE→Ratio→Clip) → 很不稳定
#   3. Reward Hacking → 模型会钻奖励模型的漏洞
#   4. 超参数极其敏感 → 调参是"玄学"
#   5. 采样效率低 → 旧数据只能复用几次
# ===========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class ValueHead(nn.Module):
    """
    Critic 的价值头 (Value Head)。
    
    它像一个"预测官"：接收语言模型的 hidden_state，
    输出一个标量，预测"在当前状态下，未来能拿多少分"。
    
    结构非常简单: 就是一个两层的 MLP (全连接网络)
    hidden_state → Linear → ReLU → Linear → 标量
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # 第一层: 保持维度
        self.fc2 = nn.Linear(hidden_size, 1)              # 第二层: 降到标量

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        参数:
            hidden_states: 语言模型最后一层的输出 [batch, seq_len, hidden_size]
        返回:
            value: 预测的价值 [batch]
        """
        # 取最后一个 token 的 hidden state 作为整个序列的表示
        x = hidden_states[:, -1, :]  # [batch, hidden_size]
        x = F.relu(self.fc1(x))       # 非线性变换
        value = self.fc2(x).squeeze(-1)  # [batch]
        return value


def compute_advantages(
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple:
    """
    使用 GAE (Generalized Advantage Estimation) 计算优势函数。
    
    Advantage = "实际发生的" - "预测官预测的"
    
    为什么从后往前算？
      因为未来的奖励会影响现在的优势。
      好比考试第 10 题做对了，这也会影响我们对第 8 题的评价
      （因为第 8 题的解题思路可能为第 10 题铺了路）
    
    参数:
        values: Critic 预测的每个时间步的价值 [T]
        rewards: 每个时间步的（修正后的）奖励 [T]
        gamma: 折扣因子（0.99 = 未来的奖励打 99 折）
        lam: GAE 的平滑参数（越大 → 看得越远，方差越大）
    返回:
        advantages: 优势值 [T]
        returns: Critic 的训练目标 [T]
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    # 从最后一步往前算
    for t in reversed(range(len(rewards))):
        # 如果是最后一步，下一步的价值为 0
        next_value = values[t + 1] if t + 1 < len(values) else 0.0

        # TD Error (时序差分误差):
        # delta = 当前奖励 + 打折后的未来预期 - 当前预期
        # 正的 delta = "惊喜"（比预期好）
        # 负的 delta = "失望"（比预期差）
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE 递推: 把历史的"惊喜/失望"衰减后累加
        advantages[t] = last_gae = delta + gamma * lam * last_gae

    # 标准化优势（训练更稳定）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # returns 是 Critic 的训练目标: "你应该预测到的值"
    returns = advantages + values
    return advantages, returns


def compute_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    PPO 的截断策略损失 + 重要性采样。
    
    重要性采样的意义:
      我们用"旧模型"（pi_old）生成的数据来训练"新模型"（pi_new）。
      ratio 就是对这种"数据复用"的校正权重。
      ratio > 1 → 新模型比旧模型更喜欢这个动作
      ratio < 1 → 新模型比旧模型更不喜欢这个动作
    """
    # 重要性采样比率: pi_new / pi_old = exp(log_pi_new - log_pi_old)
    ratio = torch.exp(new_log_probs - old_log_probs)
    adv = advantages.unsqueeze(-1) if advantages.dim() < new_log_probs.dim() else advantages

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv

    token_loss = -torch.min(surr1, surr2) * mask
    per_sample_loss = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_sample_loss.mean()


def compute_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Critic (价值模型) 的损失函数。
    
    目标: 让 Critic 的预测越来越准。
    方法: 均方误差 (MSE) = (预测 - 实际)²
    """
    return F.mse_loss(values, returns)


def train_step(
    policy_model,
    value_head: ValueHead,
    optimizer,
    value_optimizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn,
    beta: float = 0.01,
    clip_range: float = 0.2,
    max_new_tokens: int = 512,
) -> Dict:
    """
    PPO 的完整训练步（RLHF 版本）。
    
    与 GRPO 的关键区别:
      1. 每道题只生成 1 个回答（不是一组）
      2. 需要 Critic (ValueHead) 来预测价值
      3. KL 惩罚被融入到 token 级的 reward 中
    
    参数:
        policy_model: PolicyModel 实例
        value_head: Critic 价值头（需要单独训练）
        optimizer: Actor 的优化器
        value_optimizer: Critic 的优化器（PPO 有两个优化器！）
        prompts: 题目列表
        ground_truths: 标准答案列表
        reward_fn: 奖励函数
        beta: KL 惩罚系数
    """
    all_policy_losses = []
    all_value_losses = []
    all_rewards = []

    for prompt, gt in zip(prompts, ground_truths):
        # --- 阶段 1: 采样 (每题只生成 1 个回答) ---
        completions = policy_model.generate_batch(
            [prompt], max_new_tokens=max_new_tokens, temperature=0.8
        )
        completion = completions[0]

        # --- 阶段 2: 打分 ---
        rewards = reward_fn([completion], [gt])
        reward_val = rewards[0]
        all_rewards.append(reward_val)

        # --- 阶段 3: 获取 log 概率 ---
        full_ids, prompt_len = policy_model.prepare_inputs([prompt], [completion])

        # Actor 的 log_probs（需要梯度）
        new_log_probs, mask = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="actor"
        )
        old_log_probs = new_log_probs.detach()

        # Reference 的 log_probs（在 GPU 1 上）
        ref_log_probs, _ = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="ref"
        )
        ref_log_probs = ref_log_probs.to(policy_model.actor_device)

        # --- 阶段 4: 构造 token-level reward ---
        # PPO-RLHF 的特殊做法:
        # 大部分 token 的 reward = 0（中间步骤没有显式奖励）
        # 只有最后一个 token 有真正的奖励（整道题做对/做错）
        comp_len = mask.shape[1]
        token_rewards = torch.zeros(comp_len, device=policy_model.actor_device)
        valid_len = int(mask[0].sum().item())
        if valid_len > 0:
            token_rewards[valid_len - 1] = reward_val

        # KL 惩罚融入 reward（InstructGPT 的做法！）
        # 每个 token 的奖励都被扣掉一个 KL 惩罚
        kl_per_token = (new_log_probs[0] - ref_log_probs[0]).detach()
        token_rewards = token_rewards - beta * kl_per_token

        # --- 阶段 5: Critic 估值 ---
        with torch.no_grad():
            actor_outputs = policy_model.model(
                input_ids=full_ids.to(policy_model.actor_device),
                output_hidden_states=True,
            )
            hidden = actor_outputs.hidden_states[-1][:, prompt_len:, :]

        values = value_head(hidden.detach()).squeeze(0)

        # --- 阶段 6: 计算 Advantage 和 Loss ---
        advantages, returns = compute_advantages(values.detach(), token_rewards)

        policy_loss = compute_policy_loss(
            new_log_probs, old_log_probs,
            advantages.unsqueeze(0), mask, clip_range
        )
        value_loss = compute_value_loss(values, returns.detach())

        all_policy_losses.append(policy_loss)
        all_value_losses.append(value_loss)

    # --- 阶段 7: 分别更新 Actor 和 Critic ---
    # PPO 有两个独立的优化器，因为 Actor 和 Critic 的目标不同
    total_policy_loss = torch.stack(all_policy_losses).mean()
    optimizer.zero_grad()
    total_policy_loss.backward()
    optimizer.step()

    total_value_loss = torch.stack(all_value_losses).mean()
    value_optimizer.zero_grad()
    total_value_loss.backward()
    value_optimizer.step()

    return {
        "policy_loss": total_policy_loss.item(),
        "value_loss": total_value_loss.item(),
        "mean_reward": sum(all_rewards) / len(all_rewards),
    }
