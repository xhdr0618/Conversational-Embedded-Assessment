#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本脚本实现了对心理学量表问题的排序算法。
使用 jieba 分词和 Word2Vec 计算文本相似度。
"""

import json
import numpy as np
import sys
import jieba
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")

def load_questions_from_json(file_path):
    """
    从指定的 JSON 文件中加载量表问题。
    
    该 JSON 文件要求结构为：
    {
        "contents": {
            "items": {
                "1": "第一个问题内容",
                "2": "第二个问题内容",
                ...
            }
        }
    }
    
    参数:
      file_path: JSON 文件的路径
      
    返回:
      questions: 按题号排序后的问题列表（字符串列表）
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("contents", {}).get("items", {})
        # 假设题号是数字字符串，进行数值排序
        sorted_keys = sorted(items.keys(), key=lambda k: int(k))
        questions = [items[k] for k in sorted_keys]
        return questions
    except Exception as e:
        print(f"加载文件 {file_path} 时出错:", e)
        return []

def preprocess_text(text):
    """对文本进行分词预处理"""
    return list(jieba.cut(text))

def train_word2vec(questions):
    """训练 Word2Vec 模型"""
    # 对所有问题进行分词
    sentences = [preprocess_text(q) for q in questions]
    # 训练 Word2Vec 模型
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_sentence_vector(model, sentence):
    """计算句子的向量表示（词向量的平均值）"""
    words = preprocess_text(sentence)
    word_vectors = []
    for word in words:
        if word in model.wv:
            word_vectors.append(model.wv[word])
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(model.vector_size)

def compute_similarity_matrix(questions):
    """
    使用 Word2Vec 和余弦相似度计算问题之间的相似度矩阵。
    
    参数：
      questions: 问题列表
      
    返回：
      sim_matrix: numpy 数组，形状 (n, n)，每个元素为两个问题之间的余弦相似度
    """
    # 训练 Word2Vec 模型
    model = train_word2vec(questions)
    
    # 计算每个问题的向量表示
    question_vectors = [get_sentence_vector(model, q) for q in questions]
    
    # 计算余弦相似度矩阵
    n = len(questions)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # 计算余弦相似度
            vec1 = question_vectors[i]
            vec2 = question_vectors[j]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim_matrix[i][j] = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim_matrix[i][j] = 0
                
    return sim_matrix

def greedy_order_from_start(sim_matrix, start_index):
    """
    从给定起点出发使用贪心算法构造一个问题排序序列。
    
    参数：
      sim_matrix: 问题相似度矩阵，形状为 (n, n)
      start_index: 起始问题的索引
      
    返回：
      order: 索引列表，表示排序后的问题顺序
      total_sim: 序列中相邻问题余弦相似度的和
    """
    n = sim_matrix.shape[0]
    order = [start_index]
    used = set(order)
    total_sim = 0.0
    current = start_index
    
    while len(used) < n:
        best_next = None
        best_sim = -1
        for j in range(n):
            if j not in used:
                sim_ij = sim_matrix[current][j]
                if sim_ij > best_sim:
                    best_sim = sim_ij
                    best_next = j
        if best_next is None:
            break
        order.append(best_next)
        total_sim += best_sim
        used.add(best_next)
        current = best_next
    
    return order, total_sim

def find_best_order(questions):
    """
    对所有问题，尝试以每个问题作为起点构造排序序列，
    并返回相邻问题相似度总和最大的最佳排序序列。
    
    参数：
      questions: 问题字符串列表
      
    返回：
      best_order: 索引列表，表示最佳排序后的问题顺序
      best_total_sim: 最优序列中相邻问题余弦相似度总和
    """
    sim_matrix = compute_similarity_matrix(questions)
    n = len(questions)
    best_total_sim = -sys.float_info.max
    best_order = None
    
    for start in range(n):
        order, total_sim = greedy_order_from_start(sim_matrix, start)
        if total_sim > best_total_sim:
            best_total_sim = total_sim
            best_order = order
    
    return best_order, best_total_sim

def main():
    # 设置量表 JSON 文件路径
    file_path = r"scales\应激及相关行为\AIASS.json"
    
    # 从文件中加载问卷问题
    questions = load_questions_from_json(file_path)
    
    if not questions:
        print("未能加载到问题，请检查文件格式或路径。")
        return
    
    print("原始问题序列：\n")
    for i, q in enumerate(questions):
        print(f"{i+1}. {q}")
    print("\n" + "="*50 + "\n")
    
    best_order, best_sim = find_best_order(questions)
    
    print("最佳排序后的问题序列：\n")
    for i, idx in enumerate(best_order):
        print(f"{i+1}. {questions[idx]}")
    print(f"\n序列中相邻问题余弦相似度之和： {best_sim:.4f}")

if __name__ == "__main__":
    main()