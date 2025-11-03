# ===========================================
# PCA Visualization for Confusing Cases
# ===========================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import json

# -----------------------------
# Step 1: 模拟或加载案件文本
# -----------------------------
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

file_path = '/data2/zhZH/ljp-llm/mydata/analysis_data/pn-lesuo-10-18-22-06_structured.jsonl'
name = 'Extortion'
a,b = readPNjsondata(file_path)

# -----------------------------
# Step 2: 加载语义编码模型 (BGE)
# -----------------------------
model = SentenceTransformer("/data/bge-large-zh")

pos_embeddings = model.encode(a, normalize_embeddings=True)
neg_embeddings = model.encode(b, normalize_embeddings=True)

# -----------------------------
# Step 3: PCA 降维
# -----------------------------
X = np.vstack([pos_embeddings, neg_embeddings])
labels = np.array(["Positive"] * len(pos_embeddings) + ["Negative"] * len(neg_embeddings))

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# -----------------------------
# Step 4: 绘图
# -----------------------------
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(8, 6))

palette = {"Positive": "#1f77b4", "Negative": "#d62728"}
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=palette, s=80, alpha=0.75, edgecolor="k")

plt.title(f"PCA Visualization of a Confusing Scenario about {name}", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Sample Type", loc="best")
plt.tight_layout()
plt.show()
from datetime import datetime
now = datetime.now()
plt.savefig(f"/data2/zhZH/ljp-llm/mydata/analysis_img/PCA-{name}-{now.strftime('%Y-%m-%d-%H-%M')}.png", dpi=300, bbox_inches='tight')
