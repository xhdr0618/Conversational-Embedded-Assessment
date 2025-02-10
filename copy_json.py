#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

def is_numeric_dict(d):
    """
    判断字典的所有键是否均能转换为浮点数，即是否为数字型键（可包含"1.1"等格式）。
    """
    if not isinstance(d, dict):
        return False
    for k in d.keys():
        try:
            float(k)
        except ValueError:
            return False
    return True

def get_numeric_dict_values(d):
    """
    对字典 d 的键按数值（浮点）排序后，返回对应的值列表。
    """
    return [d[k] for k in sorted(d.keys(), key=lambda x: float(x))]

def extract_from_value(value):
    """
    对目标值进行平铺处理：
      - 若为列表，则对列表中每个元素判断，如是字典且所有键均可转为数字，则进一步平铺；否则直接添加；
      - 若为字典且为数字字典，则平铺后返回；
      - 其它情况包装为列表返回。
    """
    if isinstance(value, list):
        result = []
        for x in value:
            if isinstance(x, dict) and is_numeric_dict(x):
                result.extend(get_numeric_dict_values(x))
            else:
                result.append(x)
        return result
    elif isinstance(value, dict):
        if is_numeric_dict(value):
            return get_numeric_dict_values(value)
        else:
            return [value]
    else:
        return [value]

def recursive_extract_items(data):
    """
    递归搜索数据结构中所有键为 "item" 或 "items"（不区分大小写）的项，
    对符合条件的值进行平铺处理，并返回所有提取项构成的列表。
    """
    items = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in ["item", "items"]:
                extracted = extract_from_value(value)
                items.extend(extracted)
            if isinstance(value, (dict, list)):
                items.extend(recursive_extract_items(value))
    elif isinstance(data, list):
        for element in data:
            if isinstance(element, (dict, list)):
                items.extend(recursive_extract_items(element))
    return items

def extract_items_from_file(file_path):
    """
    加载单个 JSON 文件，并递归提取其中所有 "item"/"items" 项的值（经过平铺处理）。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return recursive_extract_items(data)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return []

def extract_all_items(directory):
    """
    遍历指定目录下所有 .json 文件，将每个文件中提取到的项合并为一个列表返回。
    """
    all_items = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(directory, filename)
            file_items = extract_items_from_file(file_path)
            all_items.extend(file_items)
    return all_items

def flatten_any(item):
    """
    递归平铺任意嵌套结构，返回最底层的非字典、非列表元素（转换为字符串）。
    如果遇到字典，则按键排序后继续平铺；列表元素依次平铺。
    """
    if isinstance(item, dict):
        result = []
        # 尝试将所有键转换成浮点数进行排序；若失败则按默认排序
        try:
            sorted_keys = sorted(item.keys(), key=lambda x: float(x))
        except Exception:
            sorted_keys = sorted(item.keys())
        for key in sorted_keys:
            result.extend(flatten_any(item[key]))
        return result
    elif isinstance(item, list):
        result = []
        for sub in item:
            result.extend(flatten_any(sub))
        return result
    else:
        # 转换为字符串
        return [str(item)]

def process_extracted_items(items):
    """
    对提取到的项进行进一步平铺处理，确保每个最终项为单一字符串。
    如果某项本身为非字符串则递归平铺。
    """
    final_items = []
    for item in items:
        if isinstance(item, str):
            final_items.append(item)
        else:
            flattened = flatten_any(item)
            final_items.extend(flattened)
    return final_items

def save_items_to_new_json(all_items, output_file):
    """
    将所有最终提取到的项保存到新的 JSON 文件中，格式为：
    {
      "原始问题": {
         "1": "第1个提取项",
         "2": "第2个提取项",
         "3": "第3个提取项",
         ...
      }
    }
    保证最终保存的 JSON 文件只有一层（即所有文本项目都直接存储在 "原始问题" 下）。
    """
    final_items = process_extracted_items(all_items)
    result = {"原始问题": {}}
    for idx, item in enumerate(final_items, start=1):
        result["原始问题"][str(idx)] = item
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"新 JSON 文件已保存: {output_file}")
    except Exception as e:
        print(f"写入文件 {output_file} 时出错: {e}")

def main():
    source_dir = r"D:\desk\MICCAI\llm\test"               # 原始 JSON 文件所在目录
    output_file = r"D:\desk\MICCAI\llm\rewritten_items.json"  # 输出文件路径
    all_items = extract_all_items(source_dir)
    print(f"总共提取到 {len(all_items)} 个项目（未经最终平铺）")
    save_items_to_new_json(all_items, output_file)

if __name__ == "__main__":
    main()