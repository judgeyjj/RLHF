# ===========================================================================
# policy.py - 策略模型包装器（LoRA 版本）
# ===========================================================================
#
# 【LoRA 原理】
# 你说得对！LoRA 利用了权重变化矩阵的低秩特性:
#
#   原始权重矩阵: W [d × d]，比如 [4096 × 4096] = 16M 参数
#   
#   全参数微调: W' = W + ΔW
#     ΔW 也是 [4096 × 4096]，需要训练 16M 参数
#   
#   LoRA 的洞察: ΔW 其实是低秩的！可以分解为两个小矩阵的乘积：
#     ΔW = A × B
#     A: [4096 × r]  (下投影矩阵，降维)
#     B: [r × 4096]  (上投影矩阵，升维)
#     其中 r 是秩 (rank)，通常取 8 或 16
#   
#   参数量对比:
#     全参数: 4096 × 4096 = 16,777,216 (16M)
#     LoRA:   4096 × 8 + 8 × 4096 = 65,536 (64K)
#     → 只需要原来的 0.4% 参数！
#
# 【LoRA 在 RL 训练中的优势】
#   1. 显存省: 只有 LoRA 参数需要梯度和优化器状态
#   2. 速度快: 反向传播只经过小矩阵
#   3. 安全: 原始权重冻住，不会"训废"模型
#   4. 灵活: 训完后可以 merge 回原始模型，也可以单独保存 LoRA 权重
#
# 【硬件分配（2x A40 48GB）】
#   GPU 0: Actor 模型 (LoRA) ≈ 3GB 模型 + 极少梯度 ≈ 4GB
#   GPU 1: Reference 模型 (冻结) ≈ 3GB
#   → 剩余 ~90GB 可以用于更大的 batch_size 和 group_size！
# ===========================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List


class PolicyModel:
    """
    策略模型包装器（LoRA 版本）。
    Actor 使用 LoRA 微调，Reference 完全冻结。
    """
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        actor_device="cuda:0",
        ref_device="cuda:1",
        lora_rank=8,         # LoRA 的秩 (r)，越大 → 表达能力越强 → 参数越多
        lora_alpha=16,       # LoRA 的缩放系数。通常设为 2 * rank
        lora_dropout=0.05,   # LoRA 的 dropout 比例，防止过拟合
    ):
        """
        参数:
            model_id: HuggingFace 模型名称
            actor_device: Actor 放在哪张 GPU
            ref_device: Reference 放在哪张 GPU
            lora_rank: LoRA 秩。8=轻量，16=中等，32=较重
            lora_alpha: 缩放系数。控制 LoRA 更新对原始权重的影响幅度
                        实际缩放比例 = alpha / rank
            lora_dropout: 训练时随机丢弃一部分 LoRA 参数，防止过拟合
        """
        self.actor_device = actor_device
        self.ref_device = ref_device

        # --- 加载分词器 ---
        print(f"正在加载分词器: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 加载 Actor (策略模型) 并套上 LoRA ---
        print(f"正在加载 Actor → {actor_device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(actor_device)

        # 配置 LoRA
        # target_modules: 对哪些层加 LoRA
        # 通常选择 attention 的 q, k, v 投影层
        # 有些论文也会加 o_proj 和 gate/up/down_proj (MLP 层)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,       # 因果语言建模任务
            r=lora_rank,                         # 秩：A 和 B 矩阵的"中间维度"
            lora_alpha=lora_alpha,               # 缩放系数
            lora_dropout=lora_dropout,           # dropout
            target_modules=[                     # 对这些层加 LoRA
                "q_proj",    # Query 投影层
                "k_proj",    # Key 投影层
                "v_proj",    # Value 投影层
                "o_proj",    # Output 投影层
            ],
            bias="none",                         # 不训练 bias
        )

        # 用 peft 包装模型：冻结原始权重 + 注入 LoRA 层
        self.model = get_peft_model(base_model, lora_config)

        # 打印可训练参数量（面试可以说出这个数字！）
        self.model.print_trainable_parameters()
        # 预期输出: trainable params: ~4M || all params: ~1.5B || trainable%: ~0.3%

        # --- 加载 Reference (参考模型) → GPU 1 ---
        # 参考模型不需要 LoRA，完全冻结
        print(f"正在加载 Reference → {ref_device}...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(ref_device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print("模型加载完成！(Actor 使用 LoRA 微调)")

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        批量生成回答（采样步骤）。
        
        LoRA 模型的 generate 和普通模型完全一样，
        peft 已经在内部自动处理了 W + A×B 的计算。
        """
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.actor_device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,               # 采样模式（RL 探索需要随机性）
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        completions = self.tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        return completions

    def get_log_probs_and_mask(self, full_token_ids: torch.Tensor, prompt_len: int, device: str = None):
        """
        核心函数：计算 completion 部分每个 token 的 log 概率。
        所有 RL 算法（PPO/DPO/GRPO/DAPO/GSPO）都依赖这个函数。
        
        参数:
            full_token_ids: [batch, total_seq_len] 完整 token ID (prompt + completion)
            prompt_len: prompt 的 token 长度
            device: "actor" → 用 GPU0 的 LoRA 模型
                    "ref"   → 用 GPU1 的参考模型
        返回:
            log_probs: [batch, comp_len] completion 部分的 token log 概率
            mask: [batch, comp_len] 有效 token 掩码
        """
        if device == "ref":
            model = self.ref_model
            target_device = self.ref_device
        else:
            model = self.model
            target_device = self.actor_device

        input_ids = full_token_ids.to(target_device)

        # 前向传播
        # LoRA 的魔法在这里: peft 自动计算 W + A×B，对外接口完全透明
        with torch.no_grad() if device == "ref" else torch.enable_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        # logits[t] 预测的是 token[t+1]
        # 所以取 [prompt_len-1 : -1] 对应 [prompt_len:] 的 labels
        completion_logits = logits[:, prompt_len - 1 : -1, :]
        completion_labels = input_ids[:, prompt_len:]

        # log_softmax + gather: 取出实际生成的 token 的概率
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=completion_labels.unsqueeze(-1)
        ).squeeze(-1)

        # mask: padding 位置为 0
        mask = (completion_labels != self.tokenizer.pad_token_id).float()

        return per_token_log_probs, mask

    def prepare_inputs(self, prompts: List[str], completions: List[str]):
        """将 prompt 和 completion 拼接编码为 token ID。"""
        prompt_tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        full_texts = [p + c for p, c in zip(prompts, completions)]
        full_tokens = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        return full_tokens["input_ids"], prompt_len

    def save_lora(self, save_path: str):
        """
        保存 LoRA 权重（只保存 A 和 B 矩阵，不保存原始模型）。
        文件极小: 1.5B 模型的 LoRA 权重只有 ~16MB。
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"LoRA 权重已保存到: {save_path}")

    def merge_and_save(self, save_path: str):
        """
        将 LoRA 权重合并回原始模型并保存完整模型。
        合并后的模型可以独立使用，不再依赖 peft 库。
        
        公式: W_final = W_original + A × B
        """
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"合并后的完整模型已保存到: {save_path}")
