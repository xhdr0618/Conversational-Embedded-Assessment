#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import requests

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-ifwanibmtyicatvjoqjvydmddvqwggzbcwlgmqschltrcxlv"  # 请替换为实际的 API Key
MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # 请确认使用正确的模型名称

# 示例的测量目标－表达变体映射表（可根据实际需要扩展）
mapping_table = {
    "自杀意念": {
        "核心词": ["死亡", "结束", "消失"],
        "直接询问": ["最近有过不想活的念头吗？"],
        "间接表达": ["你觉得未来会变好吗？", "有没有觉得累了想休息很久？"],
        "干扰选项": ["你有考虑换工作吗？", "喜欢什么运动？"]
    }
    # 可在此添加其他测量目标的映射
}

def load_scale(file_path):
    """
    读取 SCL90 心理量表文件，并返回解析后的 JSON 数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def send_message_to_model(messages):
    """
    通过 SiliconFlow deepseekr1 模型接口发送消息，并以流式方式获取回复。
    messages 是包含对话记录的列表，每个元素为 {"role": ROLE, "content": CONTENT}。
    返回模型回复的完整文本。
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {API_KEY}"
    }
    
    response = requests.post(API_URL, json=payload, headers=headers, stream=True)
    reply_text = ""
    
    if response.status_code == 200:
        response.encoding = 'utf-8'
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
                choices = data.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    text = delta.get("content") or delta.get("reasoning_content")
                    if text:
                        # 该函数同时打印并累积回复内容
                        print(text, end='', flush=True)
                        reply_text += text
            except Exception:
                continue
    else:
        print("请求失败，状态码：", response.status_code)
    
    print()  # 换行
    return reply_text

def dynamic_transform_question(original_question, mapping_table):
    """
    根据映射表对原始量表问题进行语义解构，
    若检测到问题中包含映射表中某测量目标的“核心词”，
    则利用动态对话生成模型将该问题转变为自然对话提问，
    以降低重测时被试记住原题目的可能性。
    若无匹配则返回原问题。
    """
    # 遍历映射表，判断是否存在匹配的“核心词”
    for target, mapping in mapping_table.items():
        for core_word in mapping.get("核心词", []):
            if core_word in original_question:
                # 构造转写提示：要求模型使用映射信息改写原问题
                prompt = (
                    f"你是一个心理学专家，请对以下心理测评原始问题进行改写，使其更具自然对话风格，"
                    f"同时保持对测量目标「{target}」的检测不变，避免重复测评时被试记住原题。\n\n"
                    f"【映射表信息】\n"
                    f"核心词：{mapping.get('核心词')}\n"
                    f"直接询问：{mapping.get('直接询问')}\n"
                    f"间接表达：{mapping.get('间接表达')}\n"
                    f"干扰选项：{mapping.get('干扰选项')}\n\n"
                    f"【原始问题】\n{original_question}\n\n"
                    f"请生成一个改写后的对话提问，仅输出提问内容，不含额外说明。"
                )
                conversation = [
                    {"role": "system", "content": "你是一个心理学专家，精通心理测评问答设计。"},
                    {"role": "user", "content": prompt}
                ]
                transformed = send_message_to_model(conversation)
                return transformed.strip()
    # 若未匹配任何映射，则直接返回原始问题
    return original_question

def main():
    # 读取 SCL90心理量表文件（请确认文件路径正确）
    scale_file = r"D:\desk\MICCAI\llm\SCL90.json"
    scale_data = load_scale(scale_file)
    
    # 从文件中获取问题，问题存放在 "contents" -> "items" 下（key 为 "1"~"90" 的字符串）
    items = scale_data.get("contents", {}).get("items", {})
    questions = []
    # 根据题号数字顺序整理出 90 个问题
    for key in sorted(items.keys(), key=lambda x: int(x)):
        questions.append(items[key])
    
    num_questions = len(questions)
    if num_questions != 90:
        print(f"警告：量表问题数量为 {num_questions} 个，原始 SCL90 应包含 90 个问题。")
    
    # 量表基本信息及说明（可根据实际文件内容调整）
    title = "SCL90心理量表"
    description = ("本量表用于评估个体的心理健康状况，涵盖躯体化、强迫症状、人际关系敏感、"
                   "抑郁、焦虑、敌对、恐怖、偏执、精神病性以及其它（睡眠及饮食情况）等多个维度。")
    
    print(f"欢迎参加《{title}》心理测评，该量表共包含 {num_questions} 个问题。")
    print("系统会对原有静态问题进行智能改写形成动态对话提问以降低重复测评效应。")
    print("请按提示逐题作答，每题答案回车确认。全部回答完毕后，系统将自动提交您的回答进行评分。")
    
    responses = []
    # 逐题提问：对每一道题先尝试用动态对话改写再提问
    for idx, question in enumerate(questions, start=1):
        # 对原始问题做动态转换（若与映射表匹配，则生成新提问，否则返回原题）
        dynamic_question = dynamic_transform_question(question, mapping_table)
        
        print(f"\n测评师（第{idx}题）：{dynamic_question}")
        answer = input("你的回答：").strip()
        responses.append({
            "原始问题": question,
            "动态提问": dynamic_question,
            "答案": answer
        })
    
    print("\n您已完成所有题目的回答，正在生成测评报告，请稍候...\n")
    
    # 整理所有问答记录，生成最终评测文本（此处可选择记录原始或动态问题）
    evaluation_text = f"《{title}》问答记录：\n描述：{description}\n\n"
    for i, entry in enumerate(responses, start=1):
        evaluation_text += (f"第{i}题（原始）：{entry['原始问题']}\n"
                            f"改写后提问：{entry['动态提问']}\n"
                            f"答案：{entry['答案']}\n\n")
    
    # 构造最终提示，要求大模型根据 SCL90评分规则计算总分及各维度得分并给出专业评析
    final_system_prompt = (
        "请根据以下 SCL90心理测评问答记录，依照该量表评分规则计算出总分以及各维度（躯体化、强迫症状、人际关系敏感、"
        "抑郁、焦虑、敌对、恐怖、偏执、精神病性、其他）的得分，并给出详细的专业评析和建议。请仅输出测评结果，不输出额外信息。"
    )
    
    conversation_history = [
        {"role": "user", "content": evaluation_text},
        {"role": "system", "content": final_system_prompt}
    ]
    
    print("测评报告：", end='')
    send_message_to_model(conversation_history)

if __name__ == "__main__":
    main()