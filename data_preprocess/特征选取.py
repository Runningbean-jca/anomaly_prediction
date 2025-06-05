import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === Step 1: 读取特征和标签 ===
X = pd.read_csv('detail_processed_data/tube3_enhanced_no_label.csv')
y = pd.read_csv('processed_data/tube3_label.csv')

# 假设标签文件只有一列，直接扁平化
if y.shape[1] == 1:
    y = y.iloc[:, 0]

# === Step 2: 特征标准化（可选但推荐） ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: 训练随机森林模型 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_scaled, y)

# === Step 4: 获取特征重要性并排序 ===
feature_importances = clf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# === Step 5: 打印前 20 名特征 ===
top_k = 20
print("\n🎯 Top 20 Features by Importance:")
print(importance_df.head(top_k))

# === Step 6: 可视化（可选） ===
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:top_k][::-1], importance_df['Importance'][:top_k][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 20 Important Features (Random Forest)")
plt.tight_layout()
plt.show()
