# ===========================================================================
# train.py - 统一训练入口
# ===========================================================================
# 支持 5 种算法: PPO, DPO, GRPO, DAPO, GSPO
# 训练方式: LoRA 微调 (只训练 ~0.3% 的参数)
# 分布式: DeepSpeed ZeRO-2 (梯度+优化器状态切分到多卡)
#
# 启动方式:
#   单卡: python train.py --algo grpo
#   多卡: deepspeed --num_gpus=2 train.py --algo grpo --deepspeed
# ===========================================================================

import argparse
import torch
from torch.utils.data import DataLoader

from src.models.policy import PolicyModel
from src.data.dataset import GSM8KDataset, collate_fn
from src.env.reward import math_acc_reward
from src.algorithms import grpo, dapo, gspo, ppo, dpo


def create_policy_model(args):
    """
    创建策略模型（统一入口）。
    根据是否启用 DeepSpeed 选择不同的初始化方式。
    """
    policy_model = PolicyModel(
        model_id=args.model_id,
        use_deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )

    if args.deepspeed:
        # --- DeepSpeed 初始化 ---
        # deepspeed.initialize 会做三件事:
        #   1. 把模型包装成 DeepSpeed Engine
        #   2. 创建 ZeRO-2 优化器（自动切分梯度和优化器状态）
        #   3. 设置 bf16 混合精度
        import deepspeed

        ds_engine, optimizer, _, _ = deepspeed.initialize(
            model=policy_model.model,               # 要训练的模型（带 LoRA）
            config=args.deepspeed_config,            # ZeRO-2 配置文件
            model_parameters=policy_model.model.parameters(),
        )
        # 把 DeepSpeed engine 绑定到 policy_model 上
        policy_model.init_deepspeed(ds_engine)
    else:
        # --- 普通模式: 手动创建优化器 ---
        optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=args.lr)

    return policy_model, optimizer


def train_grpo(args):
    """GRPO 训练流程"""
    policy_model, optimizer = create_policy_model(args)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 GRPO 训练 | Group Size={args.group_size} | LR={args.lr}")
    print(f"DeepSpeed: {'启用 ZeRO-2' if args.deepspeed else '未启用'}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            result = grpo.train_step(
                policy_model=policy_model,
                optimizer=optimizer,
                prompts=batch["prompts"],
                ground_truths=batch["answers"],
                reward_fn=math_acc_reward,
                group_size=args.group_size,
                beta=args.beta,
                max_new_tokens=args.max_new_tokens,
            )
            if step % args.log_every == 0:
                print(f"[Epoch {epoch+1} Step {step}] Loss={result['loss']:.4f} Reward={result['mean_reward']:.4f}")

    policy_model.save_lora(args.save_path + "/grpo_lora")


def train_dapo(args):
    """DAPO 训练流程"""
    policy_model, optimizer = create_policy_model(args)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 DAPO 训练 | Group Size={args.group_size} | 无 KL 惩罚")
    print(f"DeepSpeed: {'启用 ZeRO-2' if args.deepspeed else '未启用'}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            result = dapo.train_step(
                policy_model=policy_model,
                optimizer=optimizer,
                prompts=batch["prompts"],
                ground_truths=batch["answers"],
                reward_fn=math_acc_reward,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
            )
            if step % args.log_every == 0:
                print(f"[Epoch {epoch+1} Step {step}] Loss={result['loss']:.4f} "
                      f"Reward={result['mean_reward']:.4f} Resample={result['avg_resample']:.1f}")

    policy_model.save_lora(args.save_path + "/dapo_lora")


def train_gspo(args):
    """GSPO 训练流程"""
    policy_model, optimizer = create_policy_model(args)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 GSPO 训练 | Sequence-Level Ratio")
    print(f"DeepSpeed: {'启用 ZeRO-2' if args.deepspeed else '未启用'}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            result = gspo.train_step(
                policy_model=policy_model,
                optimizer=optimizer,
                prompts=batch["prompts"],
                ground_truths=batch["answers"],
                reward_fn=math_acc_reward,
                group_size=args.group_size,
                beta=args.beta,
                max_new_tokens=args.max_new_tokens,
            )
            if step % args.log_every == 0:
                print(f"[Epoch {epoch+1} Step {step}] Loss={result['loss']:.4f} Reward={result['mean_reward']:.4f}")

    policy_model.save_lora(args.save_path + "/gspo_lora")


def train_ppo(args):
    """PPO 训练流程"""
    policy_model, optimizer = create_policy_model(args)

    # PPO 需要额外的 Critic (ValueHead)
    base_config = policy_model.model.config if not policy_model.ds_engine \
        else policy_model.ds_engine.module.config
    hidden_size = base_config.hidden_size
    value_head = ppo.ValueHead(hidden_size).to(policy_model.actor_device)
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.lr)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 PPO 训练 | 需要 Critic (ValueHead)")
    print(f"DeepSpeed: {'启用 ZeRO-2' if args.deepspeed else '未启用'}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            result = ppo.train_step(
                policy_model=policy_model,
                value_head=value_head,
                optimizer=optimizer,
                value_optimizer=value_optimizer,
                prompts=batch["prompts"],
                ground_truths=batch["answers"],
                reward_fn=math_acc_reward,
                max_new_tokens=args.max_new_tokens,
            )
            if step % args.log_every == 0:
                print(f"[Epoch {epoch+1} Step {step}] PolicyLoss={result['policy_loss']:.4f} "
                      f"ValueLoss={result['value_loss']:.4f} Reward={result['mean_reward']:.4f}")

    policy_model.save_lora(args.save_path + "/ppo_lora")


def train_dpo(args):
    """DPO 训练流程（在线构造偏好对）"""
    policy_model, optimizer = create_policy_model(args)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 DPO 训练 | 在线采样构造偏好对")
    print(f"DeepSpeed: {'启用 ZeRO-2' if args.deepspeed else '未启用'}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            all_prompts, all_chosen, all_rejected = [], [], []

            for prompt, gt in zip(batch["prompts"], batch["answers"]):
                completions = policy_model.generate_batch(
                    [prompt] * 4, max_new_tokens=args.max_new_tokens
                )
                rewards = math_acc_reward(completions, [gt] * 4)

                chosen = [c for c, r in zip(completions, rewards) if r > 0.5]
                rejected = [c for c, r in zip(completions, rewards) if r <= 0.5]

                if chosen and rejected:
                    all_prompts.append(prompt)
                    all_chosen.append(chosen[0])
                    all_rejected.append(rejected[0])

            if not all_prompts:
                continue

            result = dpo.train_step(
                policy_model=policy_model,
                optimizer=optimizer,
                prompts=all_prompts,
                chosen_completions=all_chosen,
                rejected_completions=all_rejected,
                beta=args.beta,
            )
            if step % args.log_every == 0:
                print(f"[Epoch {epoch+1} Step {step}] Loss={result['loss']:.4f} Acc={result['accuracy']:.4f}")

    policy_model.save_lora(args.save_path + "/dpo_lora")


def main():
    parser = argparse.ArgumentParser(description="RL Alignment Training")
    parser.add_argument("--algo", type=str, default="grpo",
                        choices=["ppo", "dpo", "grpo", "dapo", "gspo"],
                        help="选择训练算法")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率 (LoRA 用 5e-5)")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="每批处理几道题")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO/DAPO/GSPO 的组大小")
    parser.add_argument("--beta", type=float, default=0.04, help="KL/DPO 温度系数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--max_samples", type=int, default=100, help="使用多少条训练数据")
    parser.add_argument("--log_every", type=int, default=1, help="每几步打印一次日志")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="LoRA 权重保存路径")

    # DeepSpeed 相关参数
    parser.add_argument("--deepspeed", action="store_true", help="启用 DeepSpeed ZeRO-2")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json",
                        help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的本地 GPU 编号 (由 DeepSpeed 自动设置)")

    args = parser.parse_args()

    algo_map = {
        "grpo": train_grpo,
        "dapo": train_dapo,
        "gspo": train_gspo,
        "ppo": train_ppo,
        "dpo": train_dpo,
    }

    print(f"算法: {args.algo.upper()}")
    print(f"模型: {args.model_id}")
    print(f"训练方式: LoRA (rank=8, alpha=16)")
    print(f"分布式: {'DeepSpeed ZeRO-2' if args.deepspeed else '单卡模式'}")
    print(f"数据: GSM8K (前 {args.max_samples} 条)")

    algo_map[args.algo](args)
    print("\n训练完成！LoRA 权重已保存。")


if __name__ == "__main__":
    main()
