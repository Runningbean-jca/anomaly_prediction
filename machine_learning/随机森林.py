import pandas as pd
import glob
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# ====================================
# 1. 加载并合并所有 day-level 数据
# ====================================
file_paths = sorted(glob.glob('day_data/tube*_day.csv'))
df_all = pd.concat([pd.read_csv(path) for path in file_paths], ignore_index=True)
df_all = df_all.sort_values(by='f0').reset_index(drop=True)

print(f"✅ 合并后的数据总行数: {len(df_all)}")

# ====================================
# 2. 只使用 top20 特征构造滑动窗口
# ====================================
top20_features = [
    'f9_max', 'f9_mean', 'f9_std', 'f9_min', 'f9_mean_diff',
    'f8_mean', 'f10_mean', 'f16_mean', 'f16_max', 'f6_std',
    'f10_min', 'f12_min', 'f10_mean_diff', 'f15_std', 'f6_mean_diff',
    'f8_std', 'f8_max', 'f10_max', 'f15_max', 'f7_mean_diff'
]

window_size = 3
X, y = [], []

for i in range(len(df_all) - window_size):
    window_data = df_all.iloc[i:i + window_size][top20_features].values.flatten()
    target = df_all.iloc[i + window_size]['anomaly']
    X.append(window_data)
    y.append(target)

X = pd.DataFrame(X)
y = pd.Series(y)

print(f"✅ 构建完成：共生成 {X.shape[0]} 条滑动窗口样本")

# ====================================
# 3. 拆分训练/测试集（原始比例）
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("📊 原始训练集 label 分布：")
print(y_train.value_counts())
print("📊 原始测试集 label 分布（保持不变）：")
print(y_test.value_counts())

# ====================================
# 4. 对训练集：0 类欠采样，1 类用 SMOTE 过采样
# ====================================
df_train = X_train.copy()
df_train['anomaly'] = y_train

df_0 = df_train[df_train['anomaly'] == 0]
df_1 = df_train[df_train['anomaly'] == 1]

# 欠采样 0 类
df_0_down = resample(df_0, replace=False, n_samples=200, random_state=42)

# 合并后用于 SMOTE
df_train_bal = pd.concat([df_0_down, df_1])
X_train_bal = df_train_bal.drop(columns=['anomaly'])
y_train_bal = df_train_bal['anomaly']

# === 填补缺失值 ===
X_train_bal.fillna(X_train_bal.mean(), inplace=True)

# 使用 SMOTE 合成少数类（1类）到 80
smote = SMOTE(sampling_strategy={1: 80}, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_bal, y_train_bal)

print("✅ 训练集平衡后分布：")
print(pd.Series(y_train_smote).value_counts())

# ====================================
# 5. 训练随机森林模型
# ====================================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight={0: 1, 1: 3}
)
rf.fit(X_train_smote, y_train_smote)

# ====================================
# 6. 预测 + 绘图评估
# ====================================
y_prob = rf.predict_proba(X_test)[:, 1]  # 获取预测为 1 的概率

# --- ROC Curve ---
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Precision-Recall Curve ---
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================
# 7. 选取最佳阈值（max F1）
# ====================================
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = pr_thresholds[best_idx]
print(f"📌 最佳阈值 (based on F1): {best_thresh:.3f}")

# ====================================
# 8. 使用最佳阈值重新评估
# ====================================
y_pred_thresh = (y_prob >= best_thresh).astype(int)
print("📊 使用最佳阈值后的评估报告：\n")
print(classification_report(y_test, y_pred_thresh))

# ====================================
# 9. 特征重要性输出
# ====================================
col_names = []
for t in range(window_size):
    for f in top20_features:
        col_names.append(f"{f}_t-{window_size - t}")

importances = pd.Series(rf.feature_importances_, index=col_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.head(20).plot(kind='barh')
plt.title('Top 20 Important Features (Sliding Window on Top20)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()