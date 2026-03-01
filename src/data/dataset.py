# dataset.py - 数据集加载与 Prompt 格式化（真实版本）

from datasets import load_dataset
from torch.utils.data import Dataset

class GSM8KDataset(Dataset):
    """
    GSM8K 数据集的 PyTorch 封装。
    每条数据包含一个数学题 (question) 和标准答案 (answer)。
    """
    def __init__(self, split="train", max_samples=None):
        print(f"正在加载 GSM8K {split} 数据集...")
        self.data = load_dataset("gsm8k", "main", split=split)
        if max_samples is not None:
            self.data = self.data.select(range(min(max_samples, len(self.data))))
        print(f"共加载 {len(self.data)} 条数据")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],                          # 包含推理过程 + #### 最终答案
            "prompt": format_prompt(item["question"]),         # 格式化后的 Prompt
        }

def format_prompt(question: str) -> str:
    """
    构造发给模型的 Prompt。
    使用 Qwen 的 ChatML 格式，引导模型使用 <think> 和 <answer> 标签。
    """
    system_prompt = (
        "你是一个数学助手。请先在 <think> 和 </think> 标签内写出你的思考过程，"
        "然后在 <answer> 和 </answer> 标签内给出最终的数字答案。"
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def collate_fn(batch):
    """
    DataLoader 的 collate 函数。将一批数据整理成字典。
    """
    return {
        "questions": [item["question"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
    }
