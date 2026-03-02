# ===========================================================================
# policy.py - 策略模型包装器（LoRA + DeepSpeed ZeRO-2 版本）
# ===========================================================================
#
# 【DeepSpeed ZeRO-2 原理】
# 训练时 GPU 显存主要被三样东西占据:
#   1. 模型参数 (Parameters)         ≈ 3GB
#   2. 梯度 (Gradients)              ≈ 3GB
#   3. 优化器状态 (Optimizer States)  ≈ 12GB (Adam 需要存 m, v, 参数备份等)
#
# ZeRO-2 把梯度和优化器状态切分到多张卡上:
#   不用 ZeRO: 每张卡存全部 → 18GB
#   ZeRO-2:   每张卡存 1/N → 约 10GB (2卡时)
#
# 【LoRA 原理】
# 权重矩阵 W [d×d] 的更新量 ΔW 是低秩的:
#   ΔW = A × B,  A: [d×r], B: [r×d],  r << d
# 只训练 A 和 B（约 0.3% 参数），原始 W 冻住
#
# 【硬件分配（2x A40 48GB, ZeRO-2）】
#   GPU 0 + GPU 1: Actor 模型 (LoRA) → DeepSpeed ZeRO-2 自动切分
#   Reference 模型: 单独放在 GPU 1 上做推理（不参与 ZeRO）
# ===========================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List


class PolicyModel:
    """
    策略模型包装器（LoRA + DeepSpeed ZeRO-2 版本）。
    Actor 使用 LoRA + DeepSpeed 分布式训练，Reference 完全冻结。
    """
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_deepspeed=False,
        ds_config=None,
        local_rank=-1,
    ):
        """
        参数:
            model_id: HuggingFace 模型名称
            lora_rank: LoRA 秩
            lora_alpha: LoRA 缩放系数
            lora_dropout: LoRA dropout
            use_deepspeed: 是否使用 DeepSpeed ZeRO-2
            ds_config: DeepSpeed 配置文件路径
            local_rank: 分布式训练中当前进程的 GPU 编号
        """
        self.use_deepspeed = use_deepspeed

        # DeepSpeed 模式下，由 DeepSpeed 管理设备分配
        if use_deepspeed:
            self.actor_device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda:0"
            # Reference 放在最后一张卡上（不参与 ZeRO 切分）
            num_gpus = torch.cuda.device_count()
            self.ref_device = f"cuda:{num_gpus - 1}"
        else:
            self.actor_device = "cuda:0"
            self.ref_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

        # --- 加载分词器 ---
        print(f"正在加载分词器: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 加载 Actor 并套上 LoRA ---
        print(f"正在加载 Actor...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )

        # LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # 如果不用 DeepSpeed，手动放到指定 GPU
        if not use_deepspeed:
            self.model = self.model.to(self.actor_device)

        # --- 加载 Reference (参考模型) ---
        # Reference 不参与 ZeRO 切分，单独放在一张卡上
        print(f"正在加载 Reference → {self.ref_device}...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(self.ref_device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # DeepSpeed engine 会在 train.py 中初始化后赋值
        self.ds_engine = None

        print("模型加载完成！")

    def init_deepspeed(self, ds_engine):
        """
        接收 DeepSpeed 初始化后的 engine。
        在 train.py 中调用 deepspeed.initialize() 后调用此方法。
        """
        self.ds_engine = ds_engine
        # DeepSpeed engine 会自动管理模型所在的设备
        self.actor_device = next(ds_engine.parameters()).device
        print(f"DeepSpeed engine 已绑定，Actor 在 {self.actor_device}")

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
        DeepSpeed 模式下使用 ds_engine.module 来访问原始模型。
        """
        # DeepSpeed 包装后，原始模型在 ds_engine.module 里
        model = self.ds_engine.module if self.ds_engine else self.model
        device = self.actor_device

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
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
        计算 completion 部分每个 token 的 log 概率。
        所有 RL 算法都依赖这个函数。
        """
        if device == "ref":
            model = self.ref_model
            target_device = self.ref_device
        else:
            # DeepSpeed 模式下用 ds_engine，否则用 self.model
            model = self.ds_engine.module if self.ds_engine else self.model
            target_device = self.actor_device

        input_ids = full_token_ids.to(target_device)

        # 前向传播
        with torch.no_grad() if device == "ref" else torch.enable_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        # logits[t] 预测 token[t+1]
        completion_logits = logits[:, prompt_len - 1 : -1, :]
        completion_labels = input_ids[:, prompt_len:]

        # log_softmax + gather
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=completion_labels.unsqueeze(-1)
        ).squeeze(-1)

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
        """保存 LoRA 权重（约 16MB）。"""
        model = self.ds_engine.module if self.ds_engine else self.model
        model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"LoRA 权重已保存到: {save_path}")

    def merge_and_save(self, save_path: str):
        """将 LoRA 合并回原始模型并保存（W_final = W + A×B）。"""
        model = self.ds_engine.module if self.ds_engine else self.model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"合并后的模型已保存到: {save_path}")
