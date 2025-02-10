#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json

def main():
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",  # 根据实际情况替换成您的模型名称
        "messages": [
            {"role": "user", "content": "请问 SiliconFlow deepseekr1 模型可以如何应用？"}
        ],
        "stream": True  # 开启流式输出
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        # 将 "your-api-key" 替换为您实际的 API Key
        "authorization": "Bearer sk-ifwanibmtyicatvjoqjvydmddvqwggzbcwlgmqschltrcxlv"
    }
    
    # 发送流式请求（注意设置 stream=True）
    response = requests.post(url, json=payload, headers=headers, stream=True)
    
    if response.status_code == 200:
        # 设置响应编码为 UTF-8
        response.encoding = 'utf-8'
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            # 去除前缀 "data:"（如果存在）
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                data = json.loads(line)
                choices = data.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    # 尝试获取 content 或 reasoning_content 字段
                    text = delta.get("content") or delta.get("reasoning_content")
                    if text:
                        # 使用 print 输出时确保不因解码错误导致乱码（errors='replace'）
                        print(text.encode('utf-8', errors='replace').decode('utf-8'), end='', flush=True)
            except Exception:
                continue
    # 不打印其他输出

if __name__ == "__main__":
    main() 