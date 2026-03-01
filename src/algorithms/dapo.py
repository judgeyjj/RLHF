# ===========================================================================
# dapo.py - Direct Advantage Policy Optimization（直接优势策略优化）
# ===========================================================================
#
# 【面试核心理解】
# DAPO = GRPO + 三个针对性"手术"：
#
#   手术1: 去除 KL 惩罚 → 连参考模型都不需要了！
#     GRPO 问题: KL 惩罚让模型越来越保守（熵坍塌），不敢探索新解法
#     DAPO 方案: 直接砍掉 KL 惩罚
#
#   手术2: 动态采样 (Dynamic Sampling)
#     GRPO 问题: 如果一组答案全对或全错，Advantage 全为 0，没有训练信号
#     DAPO 方案: 重新采样直到组内"有对有错"
#
#   手术3: 解耦截断 (Decoupled Clipping)
#     GRPO 问题: clip(ratio, 0.8, 1.2) 上下对称，对好回答太保守
#     DAPO 方案: clip(ratio, 0.8, 1.28) 上界更大，鼓励好回答
#
# 缺点：
#   1. 没有 KL 拉住，模型容易奖励欺诈（钻 reward 的漏洞）
#   2. 动态采样增加训练时间（简单题/极难题都需要反复采）
#   3. 仅适用于可验证奖励场景（数学题√，写诗×）
# ===========================================================================

import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    组内标准化优势（与 GRPO 一样）。
    
    参数:
        rewards: [group_size] 一组回答的奖励，通常只有 0.0 和 1.0
    """
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + 1e-8)


def is_group_valid(rewards: torch.Tensor) -> bool:
    """
    DAPO 动态采样的核心判断：这一组回答是否"有效"？
    
    "有效"的定义：组内既有做对的，也有做错的。
    如果全对或全错，这组数据对训练没有帮助（无法区分好坏）。
    
    例子:
      [1.0, 0.0, 1.0, 0.0] → max=1.0, min=0.0, max≠min → 有效！
      [1.0, 1.0, 1.0, 1.0] → max=1.0, min=1.0, max=min  → 无效！重采！
      [0.0, 0.0, 0.0, 0.0] → 同上 → 无效！重采！
    """
    return rewards.max().item() != rewards.min().item()


def compute_dapo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
) -> torch.Tensor:
    """
    DAPO 的解耦截断损失 (Decoupled Clipping)。
    
    与 GRPO/PPO 的关键区别:
      GRPO: clip(ratio, 1 - 0.2,  1 + 0.2)   → [0.8, 1.2]  对称
      DAPO: clip(ratio, 1 - 0.2,  1 + 0.28)  → [0.8, 1.28] 不对称！
            上界更大 → 对好回答更宽容，允许模型大幅增加好回答的概率
            下界正常 → 对坏回答依然严厉惩罚
    
    参数:
        new_log_probs: 当前策略的 token 级 log 概率 [group_size, comp_len]
        old_log_probs: 旧策略的 token 级 log 概率 [group_size, comp_len]
        advantages: 组内优势 [group_size]
        mask: 有效 token 掩码 [group_size, comp_len]
        clip_low: 下界截断范围（0.2 → ratio 最低 0.8）
        clip_high: 上界截断范围（0.28 → ratio 最高 1.28）
    """
    # 步骤 1: 计算重要性采样比率（和 GRPO/PPO 一样）
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 步骤 2: 广播 advantage 到 token 维度
    adv = advantages.unsqueeze(-1)

    # 步骤 3: 解耦截断 —— 关键区别在这里！
    # 下界 1 - clip_low = 0.8, 上界 1 + clip_high = 1.28
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * adv

    # 步骤 4: 应用 mask，按每条回答取平均
    token_loss = -torch.min(surr1, surr2) * mask
    per_sample_loss = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return per_sample_loss.mean()


def train_step(
    policy_model,
    optimizer,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn,
    group_size: int = 4,
    max_resample: int = 8,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    max_new_tokens: int = 512,
) -> Dict:
    """
    DAPO 的完整训练步。
    
    与 GRPO 的三个区别:
      1. 没有 ref_model 参与 loss 计算（不用参考模型！没有 KL 惩罚！）
      2. 有动态采样（全对/全错就重新采样，直到组内有区分度）
      3. 解耦截断（clip_low ≠ clip_high，上界更宽松）
    """
    all_losses = []
    all_rewards = []
    resample_counts = []  # 记录每道题采样了几轮

    for prompt, gt in zip(prompts, ground_truths):

        # --- 阶段 1: 动态采样 (Dynamic Sampling) ---
        # DAPO 独有！如果一组全对或全错，就重新采样
        completions = None
        rewards_tensor = None

        for attempt in range(max_resample):
            # 生成一组回答
            completions = policy_model.generate_batch(
                [prompt] * group_size,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
            )
            # 打分
            rewards = reward_fn(completions, [gt] * group_size)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # 检查这组是否有效（有做对的也有做错的）
            if is_group_valid(rewards_tensor):
                resample_counts.append(attempt + 1)
                break
            # 如果无效（全对或全错），继续采样
        else:
            # 达到最大重采次数仍无效，跳过这道题
            resample_counts.append(max_resample)
            continue

        all_rewards.extend(rewards)

        # --- 阶段 2: 计算优势 ---
        advantages = compute_group_advantages(rewards_tensor)

        # --- 阶段 3: 计算 log_probs ---
        full_ids, prompt_len = policy_model.prepare_inputs(
            [prompt] * group_size, completions
        )
        new_log_probs, mask = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="actor"
        )
        old_log_probs = new_log_probs.detach()

        # --- 阶段 4: 计算 Loss ---
        # 注意：DAPO 没有 KL 惩罚！这是和 GRPO 的最大区别之一
        advantages = advantages.to(policy_model.actor_device)
        loss = compute_dapo_loss(
            new_log_probs, old_log_probs, advantages, mask, clip_low, clip_high
        )
        all_losses.append(loss)

    # --- 阶段 5: 反向传播 ---
    if all_losses:
        batch_loss = torch.stack(all_losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss_val = batch_loss.item()
    else:
        loss_val = 0.0

    return {
        "loss": loss_val,
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "avg_resample": sum(resample_counts) / max(len(resample_counts), 1),
    }
