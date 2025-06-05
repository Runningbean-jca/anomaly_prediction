import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 自定义 ABOF 实现
# -----------------------------
def compute_abof(X, k=20):
    n_samples = X.shape[0]
    abof_scores = np.zeros(n_samples)

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    for i in range(n_samples):
        xi = X[i]
        neighbors = X[indices[i][1:]]  # 排除自身

        angles = []
        for j in range(len(neighbors)):
            for l in range(j+1, len(neighbors)):
                v1 = neighbors[j] - xi
                v2 = neighbors[l] - xi
                cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(cosine, -1.0, 1.0))
                angles.append(angle ** 2)

        abof_scores[i] = np.var(angles) if angles else 0.0

    return abof_scores

# -----------------------------
# 主流程
# -----------------------------

# 1. 加载数据
df = pd.read_csv('Data/tube5.csv')

# 2. 特征选择并标准化（使用 f3–f16）
feature_cols = [f'f{i}' for i in range(3, 17)]
X = df[feature_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. UMAP 降维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 4. 计算 ABOF 分数
abof_scores = compute_abof(X_scaled, k=20)
df['abof_score'] = abof_scores

# 5. 根据 ABOF 分数选取异常点（越小越异常）
threshold = np.percentile(abof_scores, 1)  # 取前1%最小的为异常
df['abof_outlier'] = (df['abof_score'] < threshold) & (df['f17'] == 0) & (df['f18'] == 0)

# 6. 标注真实异常（Tube 异常）
df['true_anomaly'] = (df['f17'] == 1) | (df['f18'] == 1)

# 7. 可视化
plt.figure(figsize=(10, 7))

# 正常点
plt.scatter(X_umap[~df['abof_outlier'], 0], X_umap[~df['abof_outlier'], 1],
            c='lightgray', s=6, label='Normal')

# ABOF 异常点
plt.scatter(X_umap[df['abof_outlier'], 0], X_umap[df['abof_outlier'], 1],
            c='red', s=10, label='ABOF Outlier')

# 真正 Tube 异常（绿色边框圈出）
plt.scatter(X_umap[df['true_anomaly'], 0], X_umap[df['true_anomaly'], 1],
            facecolors='none', edgecolors='green', s=50, label='Tube True Anomaly')

plt.title("UMAP Projection with ABOF Outlier Detection (f3–f16) for tube5.csv")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()