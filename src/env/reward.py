# reward.py - 奖励函数
# 这是 RL 训练中最关键的组件之一：告诉模型什么是"好的回答"
import re

def extract_answer_from_model(text: str) -> str:
    """
    从模型生成的冗长文本中，精确提取出最终的答案。
    """
    # 步骤 1：检查模型是否遵循了我们给出的 <answer> 标签约定
    if "<answer>" in text and "</answer>" in text:
        # 使用 split 进行切分
        # 先切掉 <answer> 之前的内容，取第 [1] 个索引（就是标签后面的内容）
        after_tag = text.split("<answer>")[1]
        # 再对剩下的内容切掉 </answer> 之后的部分，取第 [0] 个索引（就是标签中间的内容）
        answer = after_tag.split("</answer>")[0]
        # 移除两端的空格并返回
        return answer.strip()
    
    # 步骤 2：如果模型没用标签（备选方案），我们尝试提取文本中出现的最后一个数字
    # 正则表达式 r"-?\d+(?:\.\d+)?" 可以匹配正负整数及小数
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        # 使用 [-1] 索引取列表中最后一个找到的数字，这通常是数学题的答案
        return numbers[-1]
    
    # 如果实在找不到，就返回清理过的原始文本
    return text.strip()

def extract_answer_from_gsm8k(text: str) -> str:
    """
    从 GSM8K 数据集的标准答案字段中提取纯数字答案。
    标准格式通常是：'步骤描述... #### 10'
    """
    # 检查是否存在分割符 ####
    if "####" in text:
        # 使用 [-1] 取最后一部分，这样即便前面也有 #### 符号也能保证取到最后的数字
        return text.split("####")[-1].strip()
    # 如果没有分割符，直接返回清理过的文本
    return text.strip()

def math_acc_reward(completions: list[str], ground_truths: list[str]) -> list[float]:
    """
    计算批量生成的回答与标准答案的匹配程度。
    
    参数:
        completions: 模型生成的文本回答列表
        ground_truths: 对应的标准答案列表
    返回:
        奖励值列表 (1.0 代表做对了, 0.0 代表做错了)
    """
    rewards = []
    # 使用 zip 将两个列表配对，逐个取出模型回答(completion)和正确答案(gt)
    for completion, gt in zip(completions, ground_truths):
        # 提取模型给出的答案
        extracted_model = extract_answer_from_model(completion)
        # 提取数据集中的标准答案
        extracted_gt = extract_answer_from_gsm8k(gt)
        
        # 步骤 3：进行数值比较。
        # 这里使用 try...except 是为了防止模型输出的不是数字导致 float() 报错
        try:
            # 转换为浮点数比较，可以处理 "10" 跟 "10.0" 这种数学上相等的情况
            if float(extracted_model) == float(extracted_gt):
                rewards.append(1.0) # 奖励 1 分
            else:
                rewards.append(0.0) # 得 0 分
        except:
            # 如果转换失败（比如模型输出了文字答案），则进行简单的字符串直接对比
            if extracted_model == extracted_gt:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards