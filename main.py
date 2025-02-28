#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本整合了间接表达生成、筛选、排序和心理医生评估的完整流程。
"""

import json
import time
import requests
from typing import List, Dict, Tuple
import numpy as np
import sys
from generate_indirect_expressions import generate_indirect_expressions
from filter_expressions import select_best_expressions
from rank import find_best_order, compute_similarity_matrix

# API 配置
API_KEY = "sk-ifwanibmtyicatvjoqjvydmddvqwggzbcwlgmqschltrcxlv"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Pro/deepseek-ai/DeepSeek-V3"

def send_message_to_model(messages, retries=3, initial_timeout=60):
    """调用 DeepSeek V3 API"""
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
            wait_time = (2 ** attempt) + 1
            if attempt > 0:
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=initial_timeout)
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                print(f"请求频率过高，正在重试...（{attempt + 1}/{retries}）")
            else:
                print(f"API 调用失败, 状态码：{response.status_code}")
                
        except requests.exceptions.Timeout:
            print("请求超时")
        except Exception as e:
            print(f"请求错误：{e}")
            
        attempt += 1
        time.sleep(2)  # 基础等待时间
    
    return ""

def select_final_expression(filtered_expressions: Dict) -> Dict:
    """从每个问题的三个最佳表达中选择一个最终表达"""
    result = {}
    
    prompt_template = """
    作为一位专业的心理咨询师，请从以下三个间接表达中选择一个最适合用于心理测评的表达。
    这个表达应该最好地平衡了间接性和测量目标。

    原始问题："{original_question}"

    三个候选表达：
    1. {expr1}
    2. {expr2}
    3. {expr3}

    请只返回你选择的表达编号（1、2或3）。
    """
    
    for question_id, content in filtered_expressions.items():
        original_question = content["原始问题"]
        expressions = content["间接表达"]
        
        if len(expressions) < 3:
            result[question_id] = expressions[0] if expressions else ""
            continue
            
        prompt = prompt_template.format(
            original_question=original_question,
            expr1=expressions[0],
            expr2=expressions[1],
            expr3=expressions[2]
        )
        
        messages = [
            {"role": "system", "content": "你是一位专业的心理咨询师。"},
            {"role": "user", "content": prompt}
        ]
        
        response = send_message_to_model(messages)
        try:
            selected_index = int(response.strip()) - 1
            if 0 <= selected_index < len(expressions):
                result[question_id] = expressions[selected_index]
            else:
                result[question_id] = expressions[0]
        except:
            result[question_id] = expressions[0]
        
        time.sleep(2)  # 请求间隔
    
    return result

def conduct_assessment(ordered_questions: List[str]) -> Tuple[List[int], List[str]]:
    """
    进行交互式心理测评，返回每个问题的得分和回答内容
    """
    scores = []
    answers = []  # 记录所有回答
    conversation_history = []
    
    system_prompt = """你是一位专业的心理咨询师，正在进行心理测评。
你需要：
1. 以温和、专业的方式提出问题
2. 仔细倾听来访者的回答
3. 根据回答评估一个1-5的分数，其中：
   1分 = 完全不符合
   2分 = 比较不符合
   3分 = 一般
   4分 = 比较符合
   5分 = 完全符合"""

    print("\n心理咨询师：你好，我是心理咨询师。接下来我会问你一些问题，请根据你的真实感受回答。")

    for i, question in enumerate(ordered_questions, 1):
        # 提出问题
        question_prompt = f"""请以专业、温和的方式提出这个问题：

问题内容：{question}

要求：
1. 自然地承接上文
2. 适当添加引导语
3. 保持问题原文不变,但不要问出问题"""

        messages = conversation_history + [{"role": "user", "content": question_prompt}]
        response = send_message_to_model(messages)
        print(f"\n心理咨询师：{response}")
        
        # 获取用户回答
        user_answer = input("\n你的回答：")
        answers.append(user_answer)
        
        # 评估回答（只获取分数）
        evaluation_prompt = f"""请根据来访者的回答，给出1-5分的评估。
只需返回分数（1-5的整数），不需要解释。

原始问题：{question}
来访者回答：{user_answer}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
        score = send_message_to_model(messages)
        
        try:
            score = int(score.strip())
            if 1 <= score <= 5:
                scores.append(score)
            else:
                scores.append(3)
        except:
            scores.append(3)
        
        print(f"\n得分：{scores[-1]}")
        print("-" * 30)
        
        time.sleep(1)

    return scores, answers

def calculate_total_score(scores: List[int]) -> Dict:
    """计算总分和统计信息"""
    total = sum(scores)
    avg = total / len(scores)
    max_possible = len(scores) * 5
    percentage = (total / max_possible) * 100
    
    return {
        "总分": total,
        "平均分": round(avg, 2),
        "题目数": len(scores),
        "最高可能分": max_possible,
        "得分率": round(percentage, 2)
    }

def display_ordered_questions(ordered_questions: List[str], original_order: Dict[str, str] = None):
    """
    展示优化后的问题顺序
    
    参数:
        ordered_questions: 优化排序后的问题列表
        original_order: 原始问题字典（可选）
    """
    print("\n优化后的问题顺序：")
    print("=" * 80)
    
    for i, question in enumerate(ordered_questions, 1):
        # 如果提供了原始顺序，尝试找出这个问题在原始顺序中的编号
        original_number = ""
        if original_order:
            for num, q in original_order.items():
                if q == question:
                    original_number = f"(原题号: {num})"
                    break
        
        print(f"\n{i}. {question} {original_number}")
    
    print("\n" + "=" * 80)
    input("\n请检查问题顺序，按回车键继续...")

def generate_analysis_prompt(questions, answers, scores, results):
    """生成分析报告的prompt"""
    # 使用普通字符串连接而不是f-string
    prompt = """作为心理咨询师，请根据以下测评结果生成一份专业的分析报告：

测评情况：
"""
    
    # 添加问题和回答信息
    for i, (q, a, s) in enumerate(zip(questions, answers, scores)):
        prompt += f"问题{i+1}: {q}\n回答: {a}\n得分: {s}\n"
    
    # 添加结果信息
    prompt += "\n总体得分：\n"
    for key, value in results.items():
        prompt += f"{key}: {value}\n"
    
    prompt += """
请从以下方面进行分析：
1. 总体心理状态评估
2. 主要表现特征
3. 需要重点关注的方面
4. 改善建议

请用专业且温和的语言撰写。"""
    
    return prompt

def main():
    # 文件路径
    original_file = r"scales\一般心理健康与行为问题\SCL_90.json"
    indirect_file = "indirect_expressions_main.json"
    filtered_file = "filtered_expressions_main.json"

    
    # 1. 生成间接表达
    print("正在生成间接表达...")
    # 加载原始量表数据
    questions = load_questionnaire(original_file)  # 使用 dialogue_path.py 中的函数
    
    # 将问题列表转换为所需的字典格式
    questions_dict = {
        str(i+1): question 
        for i, question in enumerate(questions)
    }
    
    # 生成间接表达
    indirect_expressions = {}
    for qid, question in questions_dict.items():
        expressions = generate_indirect_expressions(question)  # 生成该问题的间接表达
        indirect_expressions[qid] = {
            "原始问题": question,
            "间接表达": expressions
        }
    
    # 保存间接表达
    with open(indirect_file, "w", encoding="utf-8") as f:
        json.dump(indirect_expressions, f, ensure_ascii=False, indent=2)
    
    # 2. 筛选最佳表达
    print("\n正在筛选最佳表达...")
    filtered_expressions = select_best_expressions(indirect_expressions)
    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(filtered_expressions, f, ensure_ascii=False, indent=2)
    
    # 3. 选择最终表达
    print("\n正在选择最终表达...")
    final_expressions = select_final_expression(filtered_expressions)
    
    # 4. 问题排序
    print("\n正在优化问题顺序...")
    questions = list(final_expressions.values())
    sim_matrix = compute_similarity_matrix(questions)
    best_order, _ = find_best_order(questions)
    ordered_questions = [questions[i] for i in best_order]
    
    # 展示优化后的问题顺序
    display_ordered_questions(ordered_questions, final_expressions)
    
    # 5. 进行交互式测评
    print("\n开始进行心理测评...")
    scores, answers = conduct_assessment(ordered_questions)
    
    # 6. 计算结果并生成分析报告
    results = calculate_total_score(scores)
    print("\n测评完成！")
    print("\n量表得分：")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    print("\n正在生成分析报告...")
    analysis_prompt = generate_analysis_prompt(questions, answers, scores, results)
    
    messages = [
        {"role": "system", "content": "你是一位专业的心理咨询师。"},
        {"role": "user", "content": analysis_prompt}
    ]
    
    analysis = send_message_to_model(messages)
    
    # 7. 保存结果
    assessment_results = {
        "问题顺序": ordered_questions,
        "回答记录": answers,
        "得分": scores,
        "统计结果": results,
        "分析报告": analysis
    }
    
    with open("assessment_results.json", "w", encoding="utf-8") as f:
        json.dump(assessment_results, f, ensure_ascii=False, indent=2)
    
    print("\n测评结果已保存到 assessment_results.json")

def load_questionnaire(file_path):
    """从 JSON 文件加载问题列表"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("contents", {}).get("items", {})
        questions = [items[key] for key in sorted(items, key=lambda key: int(key))]
        return questions
    except Exception as e:
        print(f"加载问卷文件时出错: {e}")
        return []

def generate_indirect_expressions(question: str) -> List[str]:
    """为单个问题生成间接表达"""
    prompt = f"""
你是一位心理评估专家。请为以下心理测评问题生成5个间接表达。

要求：
1. 每个表达要简洁、含蓄，但要反映原始问题的测量目标；
2. 严格按以下JSON格式返回：
{{
  "间接表达": [
    "表达1",
    "表达2",
    "表达3",
    "表达4",
    "表达5"
  ]
}}
请不要返回任何额外内容。

[原始问题]: {question}
"""
    
    messages = [
        {"role": "system", "content": "你是一位专业的心理评估专家。"},
        {"role": "user", "content": prompt}
    ]
    
    response = send_message_to_model(messages)
    try:
        result = json.loads(response)
        return result["间接表达"]
    except:
        print(f"解析间接表达失败，问题：{question}")
        return [""] * 5  # 返回5个空字符串作为默认值

if __name__ == "__main__":
    main()