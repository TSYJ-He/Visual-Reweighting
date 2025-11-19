"""
奖励函数示例

这个文件展示了如何编写自定义的奖励函数。
奖励函数接收模型生成的数据，返回奖励值和额外的指标。
"""

from typing import Any


def reward_function(data_dict: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    计算单个样本的奖励
    
    Args:
        data_dict: 包含以下字段的字典
            - prompt: str, 输入提示
            - response: str, 模型生成的响应
            - ground_truth: str, 参考答案（可选）
            - images: list, 图片数据（可选）
            - videos: list, 视频数据（可选）
    
    Returns:
        reward: float, 奖励值（通常在 -1 到 1 之间）
        metrics: dict, 额外的指标（用于监控）
    """
    prompt = data_dict.get('prompt', '')
    response = data_dict.get('response', '')
    ground_truth = data_dict.get('ground_truth', '')
    
    # 初始化奖励和指标
    reward = 0.0
    metrics = {}
    
    # ============================================
    # 示例 1: 基于长度的奖励
    # ============================================
    response_length = len(response.split())
    
    # 奖励适当长度的回答
    if 20 <= response_length <= 200:
        length_reward = 0.5
    elif response_length < 20:
        length_reward = -0.5  # 惩罚过短的回答
    else:
        length_reward = -0.2  # 轻微惩罚过长的回答
    
    reward += length_reward
    metrics['response_length'] = response_length
    metrics['length_reward'] = length_reward
    
    # ============================================
    # 示例 2: 基于关键词匹配的奖励
    # ============================================
    if ground_truth:
        # 计算关键词重叠
        ground_truth_words = set(ground_truth.lower().split())
        response_words = set(response.lower().split())
        
        # 忽略停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        ground_truth_words -= stopwords
        response_words -= stopwords
        
        if len(ground_truth_words) > 0:
            overlap_ratio = len(ground_truth_words & response_words) / len(ground_truth_words)
            keyword_reward = overlap_ratio
        else:
            keyword_reward = 0.0
        
        reward += keyword_reward
        metrics['keyword_overlap'] = overlap_ratio if 'overlap_ratio' in locals() else 0
        metrics['keyword_reward'] = keyword_reward
    
    # ============================================
    # 示例 3: 基于格式的奖励
    # ============================================
    format_reward = 0.0
    
    # 检查是否以句号结尾
    if response.strip().endswith(('.', '!', '?', '。', '！', '？')):
        format_reward += 0.1
    
    # 检查是否包含合理的标点符号
    punctuation_count = sum(c in '.,!?;:。，！？；：' for c in response)
    if punctuation_count > 0:
        format_reward += 0.1
    
    reward += format_reward
    metrics['format_reward'] = format_reward
    
    # ============================================
    # 示例 4: 基于语义一致性的奖励（需要额外的模型）
    # ============================================
    # 注意：这里只是示意，实际使用需要加载语义相似度模型
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # 
    # if ground_truth:
    #     embeddings = model.encode([response, ground_truth])
    #     semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    #     semantic_reward = semantic_similarity
    #     reward += semantic_reward
    #     metrics['semantic_similarity'] = semantic_similarity
    
    # ============================================
    # 示例 5: 基于特定任务的奖励（VQA 示例）
    # ============================================
    if 'images' in data_dict and len(data_dict['images']) > 0:
        # 对于视觉问答任务，可以添加特定的奖励
        # 例如：检查答案是否提到了视觉内容
        visual_keywords = ['图片', '图像', '照片', 'image', 'picture', 'photo']
        mentions_visual = any(keyword in response.lower() for keyword in visual_keywords)
        
        if mentions_visual:
            visual_reward = 0.2
        else:
            visual_reward = 0.0
        
        reward += visual_reward
        metrics['visual_reward'] = visual_reward
    
    # ============================================
    # 归一化和裁剪
    # ============================================
    # 将奖励裁剪到合理范围
    reward = max(-1.0, min(1.0, reward))
    metrics['total_reward'] = reward
    
    return reward, metrics


# ============================================
# 批量奖励函数（可选，用于提高效率）
# ============================================
def batch_reward_function(data_list: list[dict[str, Any]]) -> tuple[list[float], dict[str, list[Any]]]:
    """
    批量计算奖励（用于提高效率）
    
    Args:
        data_list: 数据字典列表
    
    Returns:
        rewards: 奖励列表
        metrics: 指标字典，每个键对应一个列表
    """
    rewards = []
    all_metrics = {}
    
    for data_dict in data_list:
        reward, metrics = reward_function(data_dict)
        rewards.append(reward)
        
        # 收集指标
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(value)
    
    return rewards, all_metrics


# ============================================
# 使用外部 API 的奖励函数示例
# ============================================
def reward_function_with_api(data_dict: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    使用外部 API（如 GPT-4）评估响应质量
    
    注意：这会增加训练时间和成本
    """
    import os
    # from openai import OpenAI
    
    prompt = data_dict.get('prompt', '')
    response = data_dict.get('response', '')
    ground_truth = data_dict.get('ground_truth', '')
    
    # 示例：使用 GPT-4 评分（需要取消注释并配置 API）
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # 
    # evaluation_prompt = f"""
    # 评估以下回答的质量（0-10分）：
    # 
    # 问题：{prompt}
    # 回答：{response}
    # 参考答案：{ground_truth}
    # 
    # 请只返回一个数字分数（0-10）。
    # """
    # 
    # completion = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": evaluation_prompt}]
    # )
    # 
    # score = float(completion.choices[0].message.content.strip())
    # reward = (score - 5) / 5  # 归一化到 [-1, 1]
    
    # 占位符实现
    reward = 0.0
    metrics = {'api_score': 0.0}
    
    return reward, metrics


# ============================================
# 数学推理任务的奖励函数示例
# ============================================
def math_reward_function(data_dict: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    数学推理任务的奖励函数
    """
    import re
    
    response = data_dict.get('response', '')
    ground_truth = data_dict.get('ground_truth', '')
    
    # 提取数值答案
    def extract_answer(text: str) -> str:
        # 尝试找到 "答案是" 或 "answer is" 后面的数字
        patterns = [
            r'答案是[:：]?\s*([0-9\.\-]+)',
            r'answer is[:：]?\s*([0-9\.\-]+)',
            r'\\boxed\{([0-9\.\-]+)\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 如果没找到，尝试找最后一个数字
        numbers = re.findall(r'[0-9\.\-]+', text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    pred_answer = extract_answer(response)
    true_answer = extract_answer(ground_truth)
    
    # 计算奖励
    if pred_answer and true_answer:
        try:
            pred_num = float(pred_answer)
            true_num = float(true_answer)
            
            # 精确匹配
            if abs(pred_num - true_num) < 1e-6:
                reward = 1.0
            # 接近
            elif abs(pred_num - true_num) < 0.1 * abs(true_num):
                reward = 0.5
            else:
                reward = -0.5
        except:
            reward = -0.5
    else:
        reward = -1.0  # 没有找到答案
    
    # 额外奖励：检查推理步骤
    has_steps = any(marker in response for marker in ['步骤', 'step', '首先', '然后', '因此'])
    if has_steps:
        reward += 0.2
    
    metrics = {
        'correct': 1.0 if reward == 1.0 else 0.0,
        'has_answer': 1.0 if pred_answer else 0.0,
        'has_reasoning_steps': 1.0 if has_steps else 0.0,
    }
    
    return reward, metrics


# ============================================
# 选择使用哪个奖励函数
# ============================================
# MTRL 会调用名为 'reward_function' 的函数
# 如果要使用其他函数，请重命名或使用别名

# 示例：使用数学奖励函数
# reward_function = math_reward_function

# 示例：使用 API 奖励函数
# reward_function = reward_function_with_api

