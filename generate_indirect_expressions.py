#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
import time

# 请替换为实际的 API Key
API_KEY = "sk-ifwanibmtyicatvjoqjvydmddvqwggzbcwlgmqschltrcxlv"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "deepseek-ai/DeepSeek-V3"

def send_message_to_model(messages, retries=3, timeout=60):
    """
    调用 DeepSeek V3 API 接口，发送对话记录 messages 并返回回复文本。
    messages 格式为: [{ "role": "角色", "content": "内容" }, ...]
    """
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
                # 根据返回结构解析回复内容
                if "choices" in data and len(data["choices"]) > 0:
                    if "delta" in data["choices"][0]:
                        reply_text = data["choices"][0]["delta"].get("content", "")
                    else:
                        reply_text = data["choices"][0]["message"].get("content", "")
                    return reply_text.strip()
                else:
                    return ""
            else:
                print(f"API 调用失败, 状态码：{response.status_code}")
        except requests.exceptions.Timeout as e:
            print("请求超时：", e)
        except Exception as e:
            print("请求错误：", e)
        attempt += 1
        print(f"正在重试...（{attempt}/{retries}）")
        time.sleep(1)
    
    return ""

def generate_indirect_expressions(question_text):
    """
    利用 DeepSeek V3 为给定原始问题生成 5 条间接表达。
    要求大模型严格返回 JSON 格式，形如：
    {
      "间接表达": [
         "表达1",
         "表达2",
         "表达3",
         "表达4",
         "表达5"
      ]
    }
    """
    prompt = (
        "你是一个心理学专家，现有一条心理测评的问题，请为该问题生成5条简短的间接表达。要求：\n"
        "1. 每条表达都要简洁、隐晦但能反映原测量目标；\n"
        "2. 严格以 JSON 格式返回，格式如下：\n"
        '{\n  "间接表达": [\n    "表达1",\n    "表达2",\n    "表达3",\n    "表达4",\n    "表达5"\n  ]\n}\n'
        "请不要返回多余内容。\n\n"
        f"【原始问题】：{question_text}"
    )
    
    messages = [
        {"role": "system", "content": "你是一位心理测评专家，擅长用间接表达描述心理问题。"},
        {"role": "user", "content": prompt}
    ]
    
    response_text = send_message_to_model(messages)
    if not response_text:
        return []
    
    try:
        result = json.loads(response_text)
        # 确保返回结果为符合要求的字典，并且有"间接表达"键
        if "间接表达" in result and isinstance(result["间接表达"], list):
            return result["间接表达"]
        else:
            print("返回 JSON 格式不符合预期:", response_text)
            return []
    except Exception as e:
        print("JSON解析错误，对于问题:", question_text, e)
        return []

def main():
    input_file = r"D:\desk\MICCAI\llm\rewritten_items.json"
    output_file = r"D:\desk\MICCAI\llm\indirect_expressions.json"
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载文件 {input_file} 失败: {e}")
        return
    
    # 假定 rewritten_items.json 的结构为 {"1": "问题文本", "2": "问题文本", ...}
    result = {}
    total = len(data)
    print(f"总共 {total} 条原始问题，开始生成间接表达……")
    
    # 遍历所有问题，key为数字字符串（例如 "1", "2", ...）
    for key in sorted(data.keys(), key=lambda x: int(x)):
        question_text = data[key]
        print(f"正在处理第 {key} 条问题: {question_text}")
        indirect_expressions = generate_indirect_expressions(question_text)
        result[key] = {
            "原始问题": question_text,
            "间接表达": indirect_expressions
        }
        # 可适当休眠，防止请求频率过快
        # time.sleep(1)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"生成结果已保存到: {output_file}")
    except Exception as e:
        print(f"写入文件 {output_file} 时出错: {e}")

if __name__ == "__main__":
    main()