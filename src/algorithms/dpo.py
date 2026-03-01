# ===========================================================================
# dpo.py - Direct Preference Optimization（直接偏好优化）
# ===========================================================================
#
# 【面试核心理解】
# DPO 的革命性在于：它证明了 RLHF 中的 Reward Model 可以被数学推导地
# "吸收"进一个简单的交叉熵损失里，彻底跳过了复杂的 RL 训练循环。
#
# 核心公式（用代码块表达）:
#   好回答得分 = log(策略概率) - log(参考概率)
#   坏回答得分 = log(策略概率) - log(参考概率)
#   Loss = -log(sigmoid(beta * (好回答得分 - 坏回答得分)))
#
# 直觉:
#   让模型觉得好回答比坏回答好就行了。
#   当 "好回答得分 - 坏回答得分" 越大 → sigmoid 越接近 1 → loss 越接近 0
#
# 与 PPO 对比：
#   PPO:  Actor + Critic + Reward Model + Reference = 4个模型
#   DPO:  Actor + Reference = 2个模型，且不需要 RL 循环！
#
# 缺点:
#   1. 离线数据：无法探索新解法，性能受限于数据质量
#   2. 需要偏好对数据: (好回答, 坏回答) 需要人工标注或预先构造
#   3. 在数学等可验证场景下不如在线 RL（GRPO/DAPO 能持续探索）
# ===========================================================================

import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> tuple:
    """
    DPO 的核心损失函数 —— 面试必考！
    
    参数（全部是 sequence 级的 log 概率之和）:
        policy_chosen_log_probs:  策略模型看到【好回答】的 log 概率
        policy_rejected_log_probs: 策略模型看到【坏回答】的 log 概率
        ref_chosen_log_probs:    参考模型看到【好回答】的 log 概率
        ref_rejected_log_probs:  参考模型看到【坏回答】的 log 概率
        beta: 温度系数。越大→越信任偏好数据；越小→越贴近参考模型
    
    返回:
        loss: DPO 损失值
        accuracy: 好回答得分高于坏回答的比例（用于监控）
    """
    # 步骤 1: 计算"隐式奖励"
    # 好回答: 策略模型比参考模型更喜欢它多少？
    chosen_rewards = beta * (policy_chosen_log_probs - ref_chosen_log_probs)

    # 坏回答:策略模型比参考模型更喜欢它多少？
    rejected_rewards = beta * (policy_rejected_log_probs - ref_rejected_log_probs)

    # 步骤 2: 计算差值
    # 我们希望好回答的隐式奖励远大于坏回答
    logits = chosen_rewards - rejected_rewards

    # 步骤 3: 用 logsigmoid 计算损失
    # logsigmoid(x) = log(sigmoid(x))
    # 当 logits 越大（好远超坏）→ sigmoid 趋近 1 → logsigmoid 趋近 0 → loss 趋近 0
    loss = -F.logsigmoid(logits).mean()

    # 监控指标：好回答的得分是否真的高于坏回答（面试加分项）
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, accuracy


def train_step(
    policy_model,
    optimizer,
    prompts: List[str],
    chosen_completions: List[str],
    rejected_completions: List[str],
    beta: float = 0.1,
) -> Dict:
    """
    DPO 的完整训练步。
    
    注意 DPO 和其他算法的根本区别:
      GRPO/DAPO/GSPO: 在线采样 → 打分 → 算 advantage → 更新
      DPO:            直接用现成的偏好对 → 算 loss → 更新（不需要在线采样！）
    
    在数学场景中构造偏好对的方法:
      1. 先让模型生成多个回答
      2. 做对的 → chosen（好回答）
      3. 做错的 → rejected（坏回答）
    
    参数:
        policy_model: PolicyModel 实例
        optimizer: 优化器
        prompts: 题目列表
        chosen_completions: 好回答（做对的）列表
        rejected_completions: 坏回答（做错的）列表
        beta: DPO 温度系数
    """
    # --- 步骤 1: 对 chosen (好回答) 计算 log 概率 ---
    chosen_ids, prompt_len = policy_model.prepare_inputs(prompts, chosen_completions)

    # 策略模型看好回答的概率（需要梯度，因为要更新策略模型）
    policy_chosen_lp, chosen_mask = policy_model.get_log_probs_and_mask(
        chosen_ids, prompt_len, device="actor"
    )
    # 参考模型看好回答的概率（不需要梯度，参考模型是冻住的）
    ref_chosen_lp, _ = policy_model.get_log_probs_and_mask(
        chosen_ids, prompt_len, device="ref"
    )
    ref_chosen_lp = ref_chosen_lp.to(policy_model.actor_device)

    # --- 步骤 2: 对 rejected (坏回答) 计算 log 概率 ---
    rejected_ids, _ = policy_model.prepare_inputs(prompts, rejected_completions)

    # 策略模型看坏回答的概率
    policy_rejected_lp, rejected_mask = policy_model.get_log_probs_and_mask(
        rejected_ids, prompt_len, device="actor"
    )
    # 参考模型看坏回答的概率
    ref_rejected_lp, _ = policy_model.get_log_probs_and_mask(
        rejected_ids, prompt_len, device="ref"
    )
    ref_rejected_lp = ref_rejected_lp.to(policy_model.actor_device)

    # --- 步骤 3: 将 token-level log_probs 聚合为 sequence-level ---
    # DPO 需要的是整条回答的总概率，而非每个 token 的
    # 具体做法: 对每个有效 token 的 log_prob 求和
    policy_chosen_seq = (policy_chosen_lp * chosen_mask).sum(dim=-1)
    policy_rejected_seq = (policy_rejected_lp * rejected_mask).sum(dim=-1)
    ref_chosen_seq = (ref_chosen_lp * chosen_mask).sum(dim=-1)
    ref_rejected_seq = (ref_rejected_lp * rejected_mask).sum(dim=-1)

    # --- 步骤 4: 计算 DPO Loss ---
    loss, accuracy = compute_dpo_loss(
        policy_chosen_seq, policy_rejected_seq,
        ref_chosen_seq, ref_rejected_seq,
        beta=beta,
    )

    # --- 步骤 5: 反向传播 ---
    optimizer.zero_grad()   # 清空旧梯度
    loss.backward()         # 反向传播
    optimizer.step()        # 更新参数

    return {
        "loss": loss.item(),
        "accuracy": accuracy.item(),  # 好回答得分高于坏回答的比例
    }
