# train.py - 统一训练入口
# 支持 5 种算法: PPO, DPO, GRPO, DAPO, GSPO
# 硬件: 2x A40 48GB (Actor→GPU0, Reference→GPU1)

import argparse
import torch
from torch.utils.data import DataLoader

# 我们自己的模块
from src.models.policy import PolicyModel
from src.data.dataset import GSM8KDataset, collate_fn
from src.env.reward import math_acc_reward
from src.algorithms import grpo, dapo, gspo, ppo, dpo


def train_grpo(args):
    """GRPO 训练流程"""
    # 初始化模型（Actor→GPU0, Ref→GPU1）
    policy_model = PolicyModel(
        model_id=args.model_id,
        actor_device="cuda:0",
        ref_device="cuda:1",
    )
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=args.lr)

    # 加载数据
    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 GRPO 训练 | Group Size={args.group_size} | LR={args.lr}")
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


def train_dapo(args):
    """DAPO 训练流程"""
    policy_model = PolicyModel(
        model_id=args.model_id,
        actor_device="cuda:0",
        ref_device="cuda:1",
    )
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=args.lr)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 DAPO 训练 | Group Size={args.group_size} | 无 KL 惩罚")
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


def train_gspo(args):
    """GSPO 训练流程"""
    policy_model = PolicyModel(
        model_id=args.model_id,
        actor_device="cuda:0",
        ref_device="cuda:1",
    )
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=args.lr)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 GSPO 训练 | Sequence-Level Ratio")
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


def train_ppo(args):
    """PPO 训练流程"""
    policy_model = PolicyModel(
        model_id=args.model_id,
        actor_device="cuda:0",
        ref_device="cuda:1",
    )
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=args.lr)

    # PPO 需要额外的 Critic (ValueHead)
    hidden_size = policy_model.model.config.hidden_size
    value_head = ppo.ValueHead(hidden_size).to("cuda:0")
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=args.lr)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 PPO 训练 | 需要 Critic (ValueHead)")
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


def train_dpo(args):
    """
    DPO 训练流程。
    注意: DPO 需要偏好对数据，不能直接用 GSM8K 的原始格式。
    这里我们先用 GRPO 采样来构造偏好对。
    """
    policy_model = PolicyModel(
        model_id=args.model_id,
        actor_device="cuda:0",
        ref_device="cuda:1",
    )
    optimizer = torch.optim.AdamW(policy_model.model.parameters(), lr=args.lr)

    dataset = GSM8KDataset(split="train", max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"\n{'='*60}")
    print(f"开始 DPO 训练 | 使用在线采样构造偏好对")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            # 为每道题构造偏好对: 先采样，再按照对错分组
            all_prompts, all_chosen, all_rejected = [], [], []

            for prompt, gt in zip(batch["prompts"], batch["answers"]):
                completions = policy_model.generate_batch(
                    [prompt] * 4, max_new_tokens=args.max_new_tokens
                )
                rewards = math_acc_reward(completions, [gt] * 4)

                # 找到做对的和做错的
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


def main():
    parser = argparse.ArgumentParser(description="RL Alignment Training")
    parser.add_argument("--algo", type=str, default="grpo",
                        choices=["ppo", "dpo", "grpo", "dapo", "gspo"],
                        help="选择训练算法")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace 模型 ID")
    parser.add_argument("--lr", type=float, default=1e-6, help="学习率")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="每批处理几道题")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO/DAPO/GSPO 的组大小")
    parser.add_argument("--beta", type=float, default=0.04, help="KL/DPO 温度系数")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--max_samples", type=int, default=100, help="使用多少条训练数据")
    parser.add_argument("--log_every", type=int, default=1, help="每几步打印一次日志")

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
    print(f"数据: GSM8K (前 {args.max_samples} 条)")

    algo_map[args.algo](args)
    print("\n训练完成！")


if __name__ == "__main__":
    main()
