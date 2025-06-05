import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# ====================================
# 1. 加载 day-level 数据
# ====================================
file_paths = sorted(glob.glob('day_data/tube*_day.csv'))
df_all = pd.concat([pd.read_csv(path) for path in file_paths], ignore_index=True)
df_all = df_all.sort_values(by='f0').reset_index(drop=True)

print(f"✅ 合并后的数据总行数: {len(df_all)}")

# ====================================
# 2. 滑动窗口构造样本
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
# 3. 拆分训练/测试集
# ====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("📊 原始训练集 label 分布：")
print(y_train.value_counts())
print("📊 原始测试集 label 分布：")
print(y_test.value_counts())

# ====================================
# 4. 欠采样 + SMOTE
# ====================================
df_train = X_train.copy()
df_train['anomaly'] = y_train

df_0 = df_train[df_train['anomaly'] == 0]
df_1 = df_train[df_train['anomaly'] == 1]

df_0_down = resample(df_0, replace=False, n_samples=200, random_state=42)
df_train_bal = pd.concat([df_0_down, df_1])
X_train_bal = df_train_bal.drop(columns=['anomaly'])
y_train_bal = df_train_bal['anomaly']

X_train_bal.fillna(X_train_bal.mean(), inplace=True)

smote = SMOTE(sampling_strategy={1: 80}, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_bal, y_train_bal)

# ====================================
# 5. 特征标准化（SVM 必需）
# ====================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)

X_test.fillna(X_test.mean(), inplace=True)
X_test_scaled = scaler.transform(X_test)

# ====================================
# 6. 训练 SVM 模型
# ====================================
svm = SVC(
    kernel='rbf',
    probability=True,         # 需要概率输出才能绘图
    class_weight={0: 1, 1: 3},
    random_state=42
)
svm.fit(X_train_scaled, y_train_smote)

# ====================================
# 7. 模型预测概率 + 曲线评估
# ====================================
y_prob = svm.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Curve (SVM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
plt.title("Precision-Recall Curve (SVM)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ====================================
# 8. 选择最佳阈值 + 评估
# ====================================
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = pr_thresholds[best_idx]
print(f"📌 最佳阈值 (based on F1): {best_thresh:.3f}")

y_pred_thresh = (y_prob >= best_thresh).astype(int)
print("📊 使用最佳阈值后的评估报告：\n")
print(classification_report(y_test, y_pred_thresh))

# ====================================
# 9. 不支持 feature_importance_，可选使用 SHAP/LIME
print("⚠️ SVM 不直接提供特征重要性。可使用 SHAP/LIME 分析模型解释性。")