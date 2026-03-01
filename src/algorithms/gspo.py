# ===========================================================================
# gspo.py - Group Sequence Policy Optimization（组序列策略优化）
# ===========================================================================
#
# 【面试核心理解】
# GSPO 是通义千问 (Qwen 3) 团队提出的最新算法。
# 它解决了 GRPO 在 Token 级优化时的致命问题：梯度方差太大！
#
# 核心改动：将优化粒度从 Token 级别提升到 Sequence（序列）级别。
#
# GRPO 的问题：
#   每个 token 单独算一个 ratio：
#     Token1 ratio = 1.05
#     Token2 ratio = 0.72  ← 波动很大
#     Token3 ratio = 1.31
#     ...500 个 token 就有 500 个 ratio → 梯度像坐过山车
#
# GSPO 的解法：
#   先把所有 token 概率加起来变成一个序列概率，
#   整个序列只有 1 个 ratio → 梯度更稳定！
#
# 缺点：
#   1. 丢失了 Token 级的精细信号（不知道是哪个词写错了）
#   2. 长短回答的归一化方式是新超参
#   3. 算法很新（2025年），社区验证还不充分
# ===========================================================================

import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """组内标准化优势（与 GRPO 一样）。"""
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + 1e-8)


def compute_sequence_log_probs(
    per_token_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    GSPO 的核心操作：将 Token 级 log 概率聚合为 Sequence 级。
    
    数学原理：
      在概率空间中: P(整句) = P(token1) × P(token2) × ... × P(tokenN)
      在对数空间中: log P(整句) = log P(token1) + log P(token2) + ... + log P(tokenN)
      所以我们只需要对 token log_probs 求和！
    
    除以长度的原因：
      长回答有更多 token → log_prob 之和天然更小（更多负数相加）
      如果不归一化，模型会倾向于生成短回答来获得更高的概率
    
    参数:
        per_token_log_probs: 每个 token 的 log 概率 [batch_size, seq_len]
        mask: 有效 token 掩码 [batch_size, seq_len]
    返回:
        sequence_log_probs: 每条回答的序列级 log 概率 [batch_size]
    """
    # 对有效 token 的 log_prob 求和
    sequence_log_probs = (per_token_log_probs * mask).sum(dim=-1)
    # 除以有效 token 数量做归一化
    seq_lengths = mask.sum(dim=-1).clamp(min=1)
    return sequence_log_probs / seq_lengths


def compute_gspo_loss(
    new_seq_log_probs: torch.Tensor,
    old_seq_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    GSPO 的序列级截断损失。
    
    与 GRPO 的关键区别:
      GRPO: 500 个 token → 500 个 ratio → 500 次 clip → average
      GSPO: 500 个 token → 聚合成 1 个序列概率 → 1 个 ratio → 1 次 clip
    
    参数:
        new_seq_log_probs: 当前策略的序列级 log 概率 [group_size]
        old_seq_log_probs: 旧策略的序列级 log 概率 [group_size]
        advantages: 组内优势 [group_size]
        clip_range: 截断范围
    """
    # 步骤 1: 序列级重要性采样比率
    # 注意：这里 ratio 是一个标量（每条序列一个），不是 token 级的向量！
    seq_ratio = torch.exp(new_seq_log_probs - old_seq_log_probs)

    # 步骤 2: 序列级截断（结构和 PPO 一样，但操作对象完全不同）
    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages

    # 步骤 3: 取最小值
    return -torch.min(surr1, surr2).mean()


def compute_kl_penalty(
    new_seq_log_probs: torch.Tensor,
    ref_seq_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    序列级 KL 散度惩罚。
    与 GRPO 的区别: KL 也是在序列级计算的，而非 token 级。
    """
    return (new_seq_log_probs - ref_seq_log_probs).mean()


def train_step(
    policy_model,
    optimizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn,
    group_size: int = 4,
    beta: float = 0.04,
    clip_range: float = 0.2,
    max_new_tokens: int = 512,
) -> Dict:
    """
    GSPO 的完整训练步。
    
    与 GRPO 的唯一流程区别在阶段 4：
      GRPO: 直接用 token-level log_probs 算 ratio 和 clip
      GSPO: 先把 token-level log_probs 聚合成 sequence-level，再算 ratio 和 clip
    """
    all_losses = []
    all_rewards = []

    for prompt, gt in zip(prompts, ground_truths):
        # --- 阶段 1-3: 和 GRPO 完全一样 ---
        completions = policy_model.generate_batch(
            [prompt] * group_size, max_new_tokens=max_new_tokens, temperature=0.8,
        )
        rewards = reward_fn(completions, [gt] * group_size)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        all_rewards.extend(rewards)
        advantages = compute_group_advantages(rewards_tensor).to(policy_model.actor_device)

        full_ids, prompt_len = policy_model.prepare_inputs(
            [prompt] * group_size, completions
        )
        new_token_log_probs, mask = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="actor"
        )
        old_token_log_probs = new_token_log_probs.detach()
        ref_token_log_probs, _ = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="ref"
        )
        ref_token_log_probs = ref_token_log_probs.to(policy_model.actor_device)

        # --- 阶段 4: GSPO 核心差异 — Token → Sequence 聚合 ---
        # 把每个 token 的 log_prob 加起来，变成一个序列级的数值
        new_seq_log_probs = compute_sequence_log_probs(new_token_log_probs, mask)
        old_seq_log_probs = compute_sequence_log_probs(old_token_log_probs, mask)
        ref_seq_log_probs = compute_sequence_log_probs(ref_token_log_probs, mask)

        # --- 阶段 5: 序列级 Loss ---
        # 注意：这里的 ratio 和 clip 都是在 sequence 级别操作的，不是 token 级别
        policy_loss = compute_gspo_loss(new_seq_log_probs, old_seq_log_probs, advantages, clip_range)
        kl_loss = compute_kl_penalty(new_seq_log_probs, ref_seq_log_probs)
        total_loss = policy_loss + beta * kl_loss
        all_losses.append(total_loss)

    # --- 阶段 6: 反向传播 ---
    batch_loss = torch.stack(all_losses).mean()
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return {
        "loss": batch_loss.item(),
        "mean_reward": sum(all_rewards) / len(all_rewards),
    }
