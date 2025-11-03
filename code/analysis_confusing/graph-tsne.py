'''
    使用相似的罪名 组成二分类的任务，
    yes or no
    
    

'''

# =============================================
# t-SNE / PCA Visualization for Confusing Cases
# Author: ScholarGPT
# =============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json

# -----------------------------
# Step 1: Load or simulate embeddings
# -----------------------------
# 假设你已经有两个embedding矩阵：
# pos_embeddings.shape = [num_pos, dim]
# neg_embeddings.shape = [num_neg, dim]

# 这里我们用随机数据模拟
np.random.seed(42)
pos_embeddings = np.random.rand(200, 768) + 0.1   # 正例
neg_embeddings = np.random.rand(200, 768) + 0.1   # 负例（可设定相似度更高）


def readonjson(file_path):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接解析文件对象
        
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 不存在")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 格式无效 - {e}")
    except Exception as e:
        print(f"发生未知错误：{e}")
    return data

def readPNdata():
    path = '/data2/zhZH/ljp-llm/mydata/CaseItem/cailcharge_train_data'
    path_cjo = '/data2/zhZH/ljp-llm/mydata/cjo_crimerule_traindata'
    
    theft_path = '/data2/zhZH/ljp-llm/mydata/articlefol_cjo22/article_233fol.json' # 80-80
    theft_path = '/data2/zhZH/ljp-llm/mydata/articlefol_cjo22/article_234fol.json' #96-80
    theft_pat = '/data2/zhZH/ljp-llm/mydata/articlefol_cjo22/article_236fol.json' #17-17
    theft_jsons = readonjson(theft_path)
    theft_pn = theft_jsons['examples']
    p_list = [] 
    n_list = []
    
    for item in theft_pn:
        case = item['input']
        label = item['target']
        if label == 'yes':
            p_list.append(case)
        elif label == 'no':
            n_list.append(case)
    
    return p_list,n_list


import csv

def read_fact_label_csv_basic(file_path):
    """
    使用Python标准csv库读取数据
    """
    facts = []
    labels = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # 检查列是否存在
            if 'fact' not in reader.fieldnames or 'label' not in reader.fieldnames:
                print("错误: CSV文件中缺少'fact'或'label'列")
                return [], []
            
            for row in reader:
                fl = dict()
                fl['fact'] = row['fact']
                fl['label'] = row['label']
                facts.append(fl)
                
        print(f"成功读取 {len(facts)} 条数据")
        return facts, labels
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return [], []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return [], []


def readPN_csv():
    path = '/data2/zhZH/ljp-llm/mydata/CAIL2018/data/smallmodeldata/test_crime.csv'
    # 使用示例
    facts, _ = read_fact_label_csv_basic(path)
    
    theft_positve = []
    confus_negative = []
    # for item in facts:
    #     fact = item['fact']
    #     label = item['label']
    #     if label == '盗窃':

def readPNjsondata(file_path):
    
    poslist,neglist = [],[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip() # 跳过空行
            if not stripped_line:
                continue
            try:
                data = json.loads(stripped_line)
                positive_data = data['positive']
                poslist.append(positive_data['fact'])
                negative_datas = data['negatives']
                for n in negative_datas:
                    nfact = n['fact']
                    neglist.append(nfact)
                
            except json.JSONDecodeError as e:
                print(f"解析行出错: {e}\n问题行内容: {line[:100]}...")
    return poslist,neglist

def two_emb(file_path):
    # ===========================================
    # Compute semantic embeddings for case texts
    # ===========================================

    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import FlagModel
    import numpy as np
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # 假设你的案件文本如下：
    a,b = readPNjsondata(file_path)
    # a = ["case a1 content", "case a2 content", "case a3 content", "case a4 content"]
    # b = ["case b1 content", "case b2 content", "case b3 content", "case b4 content"]

    # Step 1. 加载语义编码模型（可替换为 BGE 模型）
    # 推荐中文或多语言模型，例如：
    # model_name = "BAAI/bge-large-zh" 或 "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_name = "BAAI/bge-large-zh"
    model_path = '/data/bge-large-zh'
    model = SentenceTransformer(model_path)

    # Step 2. 计算文本向量
    pos_embeddings = model.encode(a, normalize_embeddings=True)  # shape = [num_pos, dim]
    neg_embeddings = model.encode(b, normalize_embeddings=True)  # shape = [num_neg, dim]

    # Step 3. 打印维度
    print("pos_embeddings.shape:", pos_embeddings.shape)
    print("neg_embeddings.shape:", neg_embeddings.shape)
    
    return pos_embeddings, neg_embeddings


file_path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pn-fire-10-18-22-15_structured.jsonl'
pos_embeddings, neg_embeddings = two_emb(file_path)
name = 'Arson'


# 合并
X = np.vstack([pos_embeddings, neg_embeddings])
labels = np.array(["Positive"] * len(pos_embeddings) + ["Confusing Negative"] * len(neg_embeddings))

# -----------------------------
# Step 2: Dimensionality reduction
# -----------------------------
method = "tsne"   # or "pca"

if method == "tsne":
    reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
else:
    reducer = PCA(n_components=2, random_state=42)

X_reduced = reducer.fit_transform(X)

# -----------------------------
# Step 3: Plot Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid", font_scale=1.2)

palette = {"Positive": "#ecfbff", "Confusing Negative": "#f28e16"}
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1],
                hue=labels, palette=palette, alpha=0.7, s=60, edgecolor="k")

plt.title(f"Semantic Distribution of Positive cases vs Confusing Cases ({method.upper()})", fontsize=14)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title="Confusing Cases in Theft ", loc="best")
plt.tight_layout()
plt.show()
from datetime import datetime
now = datetime.now()
plt.savefig(f"/data2/zhZH/ljp-llm/mydata/analysis_img/tsne-{name}-{now.strftime('%Y-%m-%d-%H-%M')}.png", dpi=300, bbox_inches='tight')
