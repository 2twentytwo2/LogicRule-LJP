import pandas as pd
import json
import numpy as np
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def find_similar_tfidf_enhanced(positive_examples, other_examples, 
                               similarity_threshold=0.5, 
                               max_similar_per_positive=3):
    """
    改进的TF-IDF相似度匹配，增加语义距离控制
    
    参数:
    positive_examples: 正例DataFrame
    other_examples: 其他标签的DataFrame
    similarity_threshold: 相似度阈值，只有超过此值才认为是相似
    max_similar_per_positive: 每个正例最多匹配的相似负例数量
    
    返回:
    相似对列表 [(正例索引, 负例索引, 相似度), ...]
    """
    # 合并所有文本
    all_texts = list(positive_examples['fact']) + list(other_examples['fact'])
    
    # 创建TF-IDF向量器
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # 拟合和转换文本
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # 分离正例和负例的向量
    pos_vectors = tfidf_matrix[:len(positive_examples)]
    neg_vectors = tfidf_matrix[len(positive_examples):]
    
    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(pos_vectors, neg_vectors)
    
    # 为每个正例寻找相似的负例
    similar_pairs = []
    
    for i in range(len(positive_examples)):
        # 获取该正例与所有负例的相似度
        similarities = similarity_matrix[i]
        
        # 找到超过阈值的相似负例
        above_threshold_indices = np.where(similarities >= similarity_threshold)[0]
        
        if len(above_threshold_indices) == 0:
            # 没有超过阈值的相似负例，跳过这个正例
            continue
        
        # 按相似度排序（从高到低）
        sorted_indices = above_threshold_indices[np.argsort(-similarities[above_threshold_indices])]
        
        # 限制最大数量
        selected_indices = sorted_indices[:max_similar_per_positive]
        
        # 添加选中的相似对
        for idx in selected_indices:
            similarity_score = similarities[idx]
            similar_pairs.append((i, idx, similarity_score))
    
    return similar_pairs

def build_matched_dataset_only(csv_file, target_label="label 1", 
                              method="tfidf", similarity_threshold=0.5,
                              max_similar_per_positive=3):
    """
    构建只包含有匹配的正例及其相似负例的数据集
    
    参数:
    csv_file: 输入CSV文件路径
    target_label: 目标正例标签
    method: 相似度计算方法 ("tfidf" 或 "sbert")
    similarity_threshold: 相似度阈值
    max_similar_per_positive: 每个正例最多匹配的相似负例数量
    
    返回:
    包含有匹配的正例及其相似负例的数据集DataFrame
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 检查必要的列
    if 'fact' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV文件必须包含'fact'和'label'列")
    
    # 分离正例和其他标签数据
    positive_examples = df[df['label'] == target_label].copy()
    other_examples = df[df['label'] != target_label].copy()
    
    print(f"正例数量 ({target_label}): {len(positive_examples)}")
    print(f"其他标签数据数量: {len(other_examples)}")
    
    if len(positive_examples) == 0:
        raise ValueError(f"未找到标签为 '{target_label}' 的数据")
    
    if len(other_examples) == 0:
        raise ValueError("未找到其他标签的数据")
    
    # 构建相似数据集
    if method == "tfidf":
        similar_pairs = find_similar_tfidf_enhanced(
            positive_examples, other_examples, 
            similarity_threshold, max_similar_per_positive
        )
    elif method == "sbert":
        # 这里可以添加SBERT的实现，为了简洁，我们使用TF-IDF
        similar_pairs = find_similar_tfidf_enhanced(
            positive_examples, other_examples, 
            similarity_threshold, max_similar_per_positive
        )
    else:
        raise ValueError("method参数必须是 'tfidf' 或 'sbert'")
    
    # 构建最终数据集 - 只包含有匹配的正例及其相似负例
    result_data = []
    
    # 获取有匹配的正例索引
    matched_positive_indices = set([pair[0] for pair in similar_pairs])
    
    # 为每个有匹配的正例创建记录
    for pos_idx in matched_positive_indices:
        pos_row = positive_examples.iloc[pos_idx]
        
        # 添加正例记录
        result_data.append({
            'fact': pos_row['fact'],
            'label': pos_row['label'],
            'type': 'positive',
            'match_id': f"match_{pos_idx}",
            'similarity_score': 1.0
        })
        
        # 添加该正例对应的所有相似负例
        for pair in similar_pairs:
            if pair[0] == pos_idx:
                neg_idx = pair[1]
                similarity = pair[2]
                neg_row = other_examples.iloc[neg_idx]
                
                result_data.append({
                    'fact': neg_row['fact'],
                    'label': neg_row['label'],
                    'type': 'negative_similar',
                    'match_id': f"match_{pos_idx}",
                    'similarity_score': similarity,
                    'matched_positive_fact': pos_row['fact']
                })
    
    result_df = pd.DataFrame(result_data)
    
    # 计算匹配统计
    print(f"\n匹配统计:")
    print(f"  有匹配的正例数量: {len(matched_positive_indices)}")
    print(f"  总相似负例数量: {len(similar_pairs)}")
    print(f"  最终数据集大小: {len(result_df)}")
    
    if len(matched_positive_indices) > 0 and len(similar_pairs) > 0:
        avg_similarity = np.mean([pair[2] for pair in similar_pairs])
        print(f"  平均相似度: {avg_similarity:.3f}")
    
    return result_df

def save_matched_dataset_to_jsonl(df, output_file):
    """
    将有匹配的数据集保存为JSONL格式
    
    参数:
    df: 包含有匹配的数据集的DataFrame
    output_file: 输出JSONL文件路径
    """
    
    # 处理numpy类型数据，确保可以被JSON序列化
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # 按match_id分组，创建结构化的记录
    grouped_data = {}
    
    for _, row in df.iterrows():
        match_id = row['match_id']
        
        if match_id not in grouped_data:
            grouped_data[match_id] = {
                'positive': {},
                'negatives': []
            }
        
        # 处理numpy类型
        record = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                record[col] = None
            else:
                record[col] = convert_numpy_types(value)
        
        # 根据类型添加到相应位置
        if row['type'] == 'positive':
            grouped_data[match_id]['positive'] = record
        elif row['type'] == 'negative_similar':
            grouped_data[match_id]['negatives'].append(record)
    
    # 转换为列表格式
    structured_records = []
    for match_id, data in grouped_data.items():
        structured_record = {
            'match_id': match_id,
            'positive': data['positive'],
            'negatives': data['negatives'],
            'negative_count': len(data['negatives'])
        }
        structured_records.append(structured_record)
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in structured_records:
            json_line = json.dumps(record, ensure_ascii=False, indent=None)
            f.write(json_line + '\n')
    
    print(f"\n有匹配的数据集已保存为JSONL格式: {output_file}")
    print(f"共保存 {len(structured_records)} 个匹配组")
    
    return structured_records

def save_flat_dataset_to_jsonl(df, output_file):
    """
    将平铺的数据集保存为JSONL格式（每条记录一行）
    
    参数:
    df: 包含数据集的DataFrame
    output_file: 输出JSONL文件路径
    """
    
    # 处理numpy类型数据，确保可以被JSON序列化
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    records = []
    
    for _, row in df.iterrows():
        # 将行转换为字典
        record = {}
        
        for col in df.columns:
            value = row[col]
            
            # 处理NaN值
            if pd.isna(value):
                record[col] = None
            else:
                record[col] = convert_numpy_types(value)
        
        records.append(record)
    
    # 保存为JSONL格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False, indent=None)
            f.write(json_line + '\n')
    
    print(f"平铺数据集已保存为JSONL格式: {output_file}")
    print(f"共保存 {len(records)} 条记录")
    
    return records

def create_and_save_matched_dataset(input_csv, target_label="label 1", 
                                   output_jsonl="matched_dataset.jsonl",
                                   similarity_threshold=0.5,
                                   max_similar_per_positive=3):
    """
    完整流程：创建有匹配的数据集并保存为JSONL格式
    """
    # 构建有匹配的数据集
    print("正在构建有匹配的数据集...")
    matched_df = build_matched_dataset_only(
        csv_file=input_csv,
        target_label=target_label,
        similarity_threshold=similarity_threshold,
        max_similar_per_positive=max_similar_per_positive
    )
    
    if matched_df is None or len(matched_df) == 0:
        print("没有找到任何匹配的正例，无法创建数据集")
        return None
    
    # 保存结构化的JSONL文件（按匹配组）
    structured_output = output_jsonl.replace('.jsonl', '_structured.jsonl')
    structured_records = save_matched_dataset_to_jsonl(matched_df, structured_output)
    
    # 保存平铺的JSONL文件（每条记录一行）
    flat_output = output_jsonl.replace('.jsonl', '_flat.jsonl')
    flat_records = save_flat_dataset_to_jsonl(matched_df, flat_output)
    
    # 创建并保存数据集的统计信息
    stats = {
        "dataset_info": {
            "creation_date": datetime.now().isoformat(),
            "target_label": target_label,
            "similarity_threshold": similarity_threshold,
            "max_similar_per_positive": max_similar_per_positive,
            "total_matched_groups": len(structured_records),
            "total_records": len(flat_records),
            "positive_count": len(matched_df[matched_df['type'] == 'positive']),
            "negative_count": len(matched_df[matched_df['type'] == 'negative_similar'])
        }
    }
    
    # 计算相似度统计
    if len(matched_df[matched_df['type'] == 'negative_similar']) > 0:
        negative_similarities = matched_df[matched_df['type'] == 'negative_similar']['similarity_score']
        stats["similarity_stats"] = {
            "mean": float(negative_similarities.mean()),
            "min": float(negative_similarities.min()),
            "max": float(negative_similarities.max()),
            "median": float(negative_similarities.median())
        }
    
    stats_file = output_jsonl.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计信息已保存: {stats_file}")
    
    return {
        'structured_records': structured_records,
        'flat_records': flat_records,
        'stats': stats
    }

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_csv = "/data2/zhZH/ljp-llm/mydata/CAIL2018/data/smallmodeldata/test_crime.csv"  # 替换为您的CSV文件路径
    target_label = "放火"     # 替换为目标标签
    name = 'fire'
    from datetime import datetime
    now = datetime.now()
    output_jsonl = f"/data2/zhZH/ljp-llm/mydata/analysis_data/pn-{name}-{now.strftime('%m-%d-%H-%M')}.jsonl"
    
    # 创建并保存数据集
    result = create_and_save_matched_dataset(
        input_csv=input_csv,
        target_label=target_label,
        output_jsonl=output_jsonl,
        similarity_threshold=0.90,  # 调整相似度阈值
        max_similar_per_positive=5 # 每个正例最多匹配的负例数量
    )
    
    if result is not None:
        print("\n数据集创建完成!")
        print(f"匹配组数量: {result['stats']['dataset_info']['total_matched_groups']}")
        print(f"总记录数: {result['stats']['dataset_info']['total_records']}")
        
        # 显示示例记录
        if result['structured_records']:
            print("\n示例匹配组:")
            example = result['structured_records'][0]
            print(json.dumps(example, ensure_ascii=False, indent=2))