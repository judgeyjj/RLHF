# ===========================================================================
# policy.py - 策略模型包装器（真实可运行版本）
# ===========================================================================
#
# 这个文件管理整个 RL 训练中最重要的两个模型：
#
#   Actor (策略模型) → 放在 GPU 0 (第一张 A40 48GB)
#     - 正在训练的模型，负责生成回答
#     - 需要计算梯度（requires_grad=True）
#
#   Reference (参考模型) → 放在 GPU 1 (第二张 A40 48GB)
#     - 冻住的原始模型，用于计算 KL 散度
#     - 不计算梯度（requires_grad=False）
#
# 为什么分两张卡?
#   1.5B 模型在 bfloat16 下约占 3GB
#   Actor 训练时需要存储梯度和优化器状态，实际占 ~12GB
#   Ref 只做推理，只占 ~3GB
#   分开放可以最大化利用两张 A40 的显存
# ===========================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


class PolicyModel:
    """
    策略模型包装器。
    同时管理 Actor（正在训练的模型）和 Reference（冻住的参考模型）。
    """
    def __init__(
        self,
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        actor_device="cuda:0",    # Actor 放在第 1 张 A40
        ref_device="cuda:1",      # Reference 放在第 2 张 A40
    ):
        """
        参数:
            model_id: HuggingFace 上的模型名称
            actor_device: Actor 模型放在哪张 GPU
            ref_device: Reference 模型放在哪张 GPU
        """
        self.actor_device = actor_device
        self.ref_device = ref_device

        # --- 加载分词器 (Tokenizer) ---
        # 分词器负责把文字变成数字 (token ID)，和把数字变回文字
        print(f"正在加载分词器: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        # 如果模型没有定义 pad_token，就用 eos_token (结束符) 代替
        # pad_token 用于批处理时填充短句子，使所有句子等长
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 加载 Actor (策略模型) → GPU 0 ---
        # bfloat16 是一种节省显存的半精度格式（相比 float32 省一半显存）
        print(f"正在加载 Actor → {actor_device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(actor_device)

        # --- 加载 Reference (参考模型) → GPU 1 ---
        print(f"正在加载 Reference → {ref_device}...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(ref_device)

        # 冻结参考模型: 设为 eval 模式 + 关闭所有参数的梯度
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        print("模型加载完成！")

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        批量生成回答。这是 RL 中的"采样"步骤。
        
        参数:
            prompts: 题目 Prompt 列表
            max_new_tokens: 最多生成多少个新 token
            temperature: 温度系数
                         高温 (1.0+) → 回答更随机、更有创造性 (探索)
                         低温 (0.1-) → 回答更确定、更保守 (利用)
                         RL 训练中通常 0.7-1.0，需要一定的随机性来探索
            num_return_sequences: 每个 prompt 返回几个回答
        返回:
            completions: 生成的回答文本列表
        """
        # 步骤 1: 编码输入 (文字 → token ID)
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.actor_device)

        # 步骤 2: 调用 model.generate 进行自回归生成
        # do_sample=True 表示从概率分布中采样（而非总是选概率最高的 token）
        # 这是 RL 中探索的基础！
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 步骤 3: 只截取新生成的部分 (去掉 prompt 的 token)
        # 例如 prompt 有 50 个 token，生成了 100 个 token
        # 我们只要后面 50 个（回答部分）
        prompt_len = inputs["input_ids"].shape[1]
        completions = self.tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )
        return completions

    def get_log_probs_and_mask(self, full_token_ids: torch.Tensor, prompt_len: int, device: str = None):
        """
        核心函数：计算 completion 部分每个 token 的 log 概率。
        
        这是所有 RL 算法的基础！PPO、GRPO、DPO 都需要调用这个函数。
        
        参数:
            full_token_ids: [batch, total_seq_len] 完整的 token ID (prompt + completion)
            prompt_len: prompt 部分有多少个 token
                        我们只计算 completion 部分的概率，不计算 prompt 的
            device: "actor" → 用 GPU0 的策略模型 (需要梯度)
                    "ref"   → 用 GPU1 的参考模型 (不需要梯度)
        返回:
            log_probs: [batch, completion_len] 每个 completion token 的 log 概率
            mask: [batch, completion_len] 有效 token 的掩码 (1=有效, 0=padding)
        """
        # 根据 device 参数选择使用哪个模型
        if device == "ref":
            model = self.ref_model
            target_device = self.ref_device
        else:
            model = self.model
            target_device = self.actor_device

        input_ids = full_token_ids.to(target_device)

        # 前向传播得到 logits
        # logits 的形状: [batch, seq_len, vocab_size]
        # logits[i][t][v] = 模型认为位置 t 处 token v 的"原始分数"
        with torch.no_grad() if device == "ref" else torch.enable_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits

        # 关键理解: logits[t] 预测的是 token[t+1]
        # 所以我们取 [prompt_len-1 : -1] 的 logits 来对应 [prompt_len:] 的 labels
        # 例如: prompt 长 5 个 token
        #   logits[4] 预测 token[5] (completion 的第一个 token)
        #   logits[5] 预测 token[6] (completion 的第二个 token)
        #   ...
        completion_logits = logits[:, prompt_len - 1 : -1, :]  # [batch, comp_len, vocab]
        completion_labels = input_ids[:, prompt_len:]           # [batch, comp_len]

        # 计算 log 概率
        # log_softmax: 把原始 logits 转为归一化的 log 概率分布
        log_probs = torch.log_softmax(completion_logits, dim=-1)

        # gather: 从整个词表的概率分布中，"取出"实际生成的那个 token 的概率
        # 这就是之前详细讲过的 gather 操作
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=completion_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch, comp_len]

        # 构建 mask: padding 位置为 0，有效 token 位置为 1
        # 这确保我们不会计算 padding 部分的 loss
        mask = (completion_labels != self.tokenizer.pad_token_id).float()

        return per_token_log_probs, mask

    def prepare_inputs(self, prompts: List[str], completions: List[str]):
        """
        将 prompt 和 completion 拼接并编码为 token ID。
        
        例子:
            prompt = "5+3=?"
            completion = "答案是8"
            full_text = "5+3=?答案是8"
            → 编码为 [token_id_1, token_id_2, ..., token_id_N]
        
        返回:
            full_ids: [batch, total_seq_len] 完整的 token ID
            prompt_len: prompt 的 token 长度（用于分割 prompt 和 completion）
        """
        # 先编码 prompt 来获取其 token 长度
        prompt_tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        # 拼接 prompt + completion 一起编码
        full_texts = [p + c for p, c in zip(prompts, completions)]
        full_tokens = self.tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        return full_tokens["input_ids"], prompt_len
