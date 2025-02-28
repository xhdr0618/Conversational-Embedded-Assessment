#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本使用 DeepSeek V3 从心理医生的专业角度评估和筛选间接表达。
"""

import json
import requests
import time
from typing import List, Dict, Tuple

# API 配置
API_KEY = "sk-ifwanibmtyicatvjoqjvydmddvqwggzbcwlgmqschltrcxlv"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Pro/deepseek-ai/DeepSeek-V3"

def send_message_to_model(messages, retries=3, timeout=60):
    """调用 DeepSeek V3 API 接口"""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }
    
    attempt = 0
    while attempt < retries:
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"API 调用失败, 状态码：{response.status_code}")
        except requests.exceptions.Timeout:
            print("请求超时")
        except Exception as e:
            print(f"请求错误：{e}")
        attempt += 1
        print(f"正在重试...（{attempt}/{retries}）")
        time.sleep(1)
    
    return ""

def load_expressions(file_path: str) -> Dict:
    """加载间接表达数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return {}

def evaluate_expression(original_question: str, expression: str) -> float:
    """使用 DeepSeek V3 评估单个间接表达的适用性"""
    if not expression:
        return 0.0
        
    prompt = f"""作为一名专业的心理咨询师，请评估以下间接表达对原始问题的适用性：

原始问题："{original_question}"
间接表达："{expression}"

请从以下几个方面进行评估并给出0-100的总分：
1. 专业性 (0-25分)：是否符合心理咨询的专业标准
2. 共情性 (0-25分)：是否体现出对来访者的理解和共情
3. 引导性 (0-25分)：是否能够引导来访者进一步探索和表达
4. 适当性 (0-25分)：表达的方式和程度是否恰当

请只返回一个数字分数。"""

    messages = [
        {"role": "system", "content": "你是一位经验丰富的心理咨询师，专门评估心理咨询中的表达方式。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response_text = send_message_to_model(messages)
        if response_text:
            return float(response_text)
        return 0.0
    except Exception as e:
        print(f"评估过程出错: {e}")
        return 0.0

def select_best_expressions(data: Dict) -> Dict:
    """为每个原始问题选择最适合的三个间接表达"""
    result = {}
    
    for question_id, content in data.items():
        original_question = content["原始问题"]
        expressions = content["间接表达"]
        
        print(f"\n正在评估第 {question_id} 个问题的间接表达...")
        print(f"原始问题：{original_question}")
        
        # 评估每个表达
        scored_expressions = []
        for expr in expressions:
            if expr:
                score = evaluate_expression(original_question, expr)
                scored_expressions.append((expr, score))
                print(f"\n表达：{expr}")
                print(f"得分：{score}")
                # 添加短暂延时，避免请求过快
                time.sleep(0.5)
        
        # 按分数降序排序并选择前三个
        scored_expressions.sort(key=lambda x: x[1], reverse=True)
        best_three = [expr for expr, _ in scored_expressions[:3]]
        
        # 保存结果
        result[question_id] = {
            "原始问题": original_question,
            "间接表达": best_three
        }
    
    return result

def save_results(results: Dict, output_path: str):
    """将结果保存为JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存结果时出错: {e}")

def main():
    input_file = r"D:\desk\MICCAI\llm\indirect_expressions.json"
    output_file = r"D:\desk\MICCAI\llm\filtered_expressions.json"
    
    # 加载数据
    data = load_expressions(input_file)
    if not data:
        print("未能加载间接表达数据")
        return
    
    print("开始评估间接表达...")
    # 选择最佳表达
    best_expressions = select_best_expressions(data)
    
    # 保存结果
    save_results(best_expressions, output_file)
    
    # 打印汇总结果
    print("\n评估完成！每个问题的最佳三个表达已保存。")

if __name__ == "__main__":
    main()