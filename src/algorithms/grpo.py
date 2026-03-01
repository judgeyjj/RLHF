# ===========================================================================
# grpo.py - Group Relative Policy Optimization（组内相对策略优化）
# ===========================================================================
#
# 【面试核心理解】
# GRPO 是 DeepSeek-R1 的核心训练算法。
# 它的革命性在于：彻底砍掉了 PPO 中极吃显存的 Critic (价值) 模型！
#
# 核心思想：
#   PPO 需要 Critic 来估计 "这道题平时能拿多少分" (Baseline/Value)。
#   GRPO 说："我不要 Critic！我让同一道题产生一组(Group)回答，
#   用这组回答的平均分作为 Baseline！"
#
# 举例：
#   题目: "5 + 3 = ?"
#   组内生成 4 个回答 (Group size = 4):
#     回答1: "8"   → 奖励 1.0
#     回答2: "9"   → 奖励 0.0
#     回答3: "8"   → 奖励 1.0
#     回答4: "7"   → 奖励 0.0
#
#   组平均奖励 = 0.5, 组标准差 = 0.5
#   回答1 的 Advantage = (1.0 - 0.5) / 0.5 = +1.0  → 鼓励！
#   回答2 的 Advantage = (0.0 - 0.5) / 0.5 = -1.0  → 惩罚！
#
# 与 PPO 对比：
#   PPO:  Advantage = Reward - Critic预测     （需要 Critic 模型）
#   GRPO: Advantage = (Reward - 组平均) / 组标准差 （不需要 Critic！）
#
# 缺点：
#   1. 全对/全错组无训练信号（Advantage 全为 0）
#   2. KL 散度导致熵坍塌，模型输出越来越单一
#   3. Group Size 是新超参，太小不准太大太慢
# ===========================================================================

import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    GRPO 核心创新：用组内平均分代替 Critic 模型来计算 Advantage。
    
    原理：
      PPO:  advantage = reward - critic_model.predict(state)   ← 需要一个大模型
      GRPO: advantage = (reward - group_mean) / group_std       ← 只需一行数学！
    
    参数:
        rewards: [group_size] 一组回答的奖励值，例如 [1.0, 0.0, 1.0, 0.0]
    返回:
        advantages: [group_size] 标准化后的优势值，例如 [+1.0, -1.0, +1.0, -1.0]
    """
    # 计算组内平均奖励（代替 Critic 的预测值）
    mean = rewards.mean()
    # 计算标准差（用于标准化，让 advantage 的尺度一致）
    std = rewards.std()
    # 标准化：(实际得分 - 平均分) / 标准差
    # 加 1e-8 防止除零（全对或全错时 std=0）
    return (rewards - mean) / (std + 1e-8)


def compute_grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    GRPO 的策略损失函数。
    结构和 PPO 的 Clipped Loss 完全一样，区别仅在于 advantage 的来源不同。
    
    参数:
        new_log_probs: 当前策略（正在训练的模型）对每个 token 的对数概率
                       形状: [group_size, completion_length]
        old_log_probs: 采样时旧策略的对数概率（用于计算重要性采样比率）
                       形状: [group_size, completion_length]
        advantages:    组内标准化后的优势值
                       形状: [group_size]
        mask:          有效 token 掩码（1=有效，0=padding填充）
                       形状: [group_size, completion_length]
        clip_range:    PPO 截断范围，默认 0.2，即 ratio 被限制在 [0.8, 1.2]
    返回:
        loss: 标量，策略损失值
    """
    # 步骤 1: 计算重要性采样比率 (Importance Sampling Ratio)
    # ratio = π_new(token) / π_old(token)
    # 在对数空间中: ratio = exp(log_π_new - log_π_old)
    ratio = torch.exp(new_log_probs - old_log_probs)  # [group_size, comp_len]

    # 步骤 2: 将 sequence-level 的 advantage 广播到 token 维度
    # advantages 是 [group_size]，ratio 是 [group_size, comp_len]
    # unsqueeze(-1) 把 [4] 变成 [4, 1]，然后自动广播到 [4, comp_len]
    adv = advantages.unsqueeze(-1)  # [group_size, 1]

    # 步骤 3: PPO-style 截断目标
    # surr1 = 未截断版本
    surr1 = ratio * adv
    # surr2 = 截断版本：ratio 被限制在 [1-eps, 1+eps] 之间
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv

    # 步骤 4: 取两者的最小值（保守更新）
    # 乘以 mask 确保只计算有效 token，不计算 padding
    token_loss = -torch.min(surr1, surr2) * mask

    # 步骤 5: 每条回答的 loss = 有效 token 的 loss 之和 / 有效 token 数
    # 这样可以防止长回答的 loss 天然比短回答大
    per_sample_loss = token_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    # 步骤 6: 全组的平均 loss
    return per_sample_loss.mean()


def compute_kl_penalty(
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL 散度惩罚 —— 也就是"紧箍咒"。
    防止策略模型偏离参考模型太远。
    
    公式: KL ≈ log_π_new - log_π_ref  (这是近似的 KL 散度)
    
    参数:
        new_log_probs: 策略模型（正在训练的）对每个 token 的 log 概率
        ref_log_probs: 参考模型（冻住的原始模型）对每个 token 的 log 概率
        mask: 有效 token 掩码
    """
    # 计算每个 token 的 KL 散度，只在有效位置计算
    kl = (new_log_probs - ref_log_probs) * mask
    # 取所有有效 token 的平均 KL
    return kl.sum() / mask.sum().clamp(min=1)


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
    GRPO 的完整训练步骤。
    
    完整流程：
      1. 对每道题，让模型生成 group_size 个不同的回答
      2. 用 reward_fn（我们的规则函数）给每个回答打分
      3. 在组内计算 advantage（用平均分代替 Critic）
      4. 计算重要性采样比率 (ratio) + 截断损失 (clipped loss)
      5. 加入 KL 散度惩罚（防止模型跑偏）
      6. 反向传播更新 Actor 参数
    
    参数:
        policy_model: PolicyModel 实例（包含 actor 和 ref 模型）
        optimizer: PyTorch 优化器（如 AdamW）
        prompts: 一批题目的 Prompt 列表
        ground_truths: 对应的标准答案列表
        reward_fn: 奖励函数（来自 reward.py 的 math_acc_reward）
        group_size: 每道题生成几个回答（组大小），通常 4-16
        beta: KL 惩罚系数。越大 → 越保守。越小 → 越自由探索
        clip_range: PPO 截断范围
        max_new_tokens: 模型最多生成多少个 token
    """
    all_losses = []
    all_rewards = []

    # 遍历每道题
    for prompt, gt in zip(prompts, ground_truths):

        # --- 阶段 1: 组内采样 (Group Sampling) ---
        # 同一道题让模型生成 group_size 个不同的回答
        # 因为 temperature=0.8 且 do_sample=True，每次生成的回答都略有不同
        completions = policy_model.generate_batch(
            [prompt] * group_size,     # 把同一个 prompt 复制 group_size 份
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )

        # --- 阶段 2: 计算奖励 (Reward Scoring) ---
        # 用我们的规则函数给每个回答打分：做对 1.0，做错 0.0
        rewards = reward_fn(completions, [gt] * group_size)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        all_rewards.extend(rewards)

        # --- 阶段 3: 计算组内优势 (Group Advantage) ---
        # 这是 GRPO 的灵魂：用组平均分代替 Critic 的预测
        advantages = compute_group_advantages(rewards_tensor)

        # --- 阶段 4: 准备 token-level 的 log 概率 ---
        # 把 prompt 和 completion 拼接起来，编码成 token ID
        full_ids, prompt_len = policy_model.prepare_inputs(
            [prompt] * group_size, completions
        )

        # 计算当前策略模型对这些回答中每个 token 的 log 概率
        # 这一步需要梯度（因为我们要通过它来更新 Actor）
        new_log_probs, mask = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="actor"
        )

        # 旧策略的 log 概率（这里简化为 detach 当前的）
        # 真实 PPO 训练中，这应该是采样时记录下来的
        old_log_probs = new_log_probs.detach()

        # 参考模型的 log 概率（在 GPU 1 上计算，不需要梯度）
        ref_log_probs, _ = policy_model.get_log_probs_and_mask(
            full_ids, prompt_len, device="ref"
        )
        # 把 ref 的结果搬到 Actor 所在的 GPU（GPU 0）上
        ref_log_probs = ref_log_probs.to(policy_model.actor_device)

        # --- 阶段 5: 计算总 Loss ---
        advantages = advantages.to(policy_model.actor_device)

        # 策略损失：让好回答的概率上升，坏回答的概率下降
        policy_loss = compute_grpo_loss(
            new_log_probs, old_log_probs, advantages, mask, clip_range
        )
        # KL 惩罚：防止模型偏离参考模型太远
        kl_loss = compute_kl_penalty(new_log_probs, ref_log_probs, mask)

        # 总损失 = 策略损失 + KL 惩罚系数 * KL 惩罚
        total_loss = policy_loss + beta * kl_loss
        all_losses.append(total_loss)

    # --- 阶段 6: 反向传播并更新参数 ---
    batch_loss = torch.stack(all_losses).mean()  # 多道题的 loss 取平均
    optimizer.zero_grad()   # 清空上一步的梯度
    batch_loss.backward()   # 反向传播，计算每个参数的梯度
    optimizer.step()        # 用梯度更新模型参数

    return {
        "loss": batch_loss.item(),                          # 当前步的 loss
        "mean_reward": sum(all_rewards) / len(all_rewards),  # 当前步的平均奖励
    }
