import pandas as pd
import numpy as np
from keras import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. 读取原始数据 ===
df = pd.read_csv("Data/tube4.csv")  # 使用你自己的路径
feature_cols = [f"f{i}" for i in range(3, 16)]  # f1是时间，f17/f18是异常标签
X = df[feature_cols].copy()

# === 2. 标准化特征 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. 构建 AutoEncoder 模型 ===
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(12, activation='relu')(input_layer)
encoded = Dense(6, activation='relu')(encoded)
decoded = Dense(12, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# === 4. 模型训练 ===
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
autoencoder.fit(X_train, X_train,
                epochs=50, batch_size=32, shuffle=True,
                validation_data=(X_val, X_val), verbose=1)

# === 5. 重构误差 + 可视化 ===
X_pred = autoencoder.predict(X_scaled)
recon_error = np.mean((X_scaled - X_pred) ** 2, axis=1)

plt.figure(figsize=(8, 5))
sns.histplot(recon_error, bins=50, kde=True)
plt.title("Autoencoder Reconstruction Error")
plt.xlabel("Reconstruction MSE")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# === 6. 设置阈值识别异常点 ===
threshold = np.percentile(recon_error, 95)
outlier_mask = recon_error > threshold

# === 7. 过滤掉真实异常标签 (f17 or f18 == 1) ===
true_anomaly_mask = (df['f17'] == 1) | (df['f18'] == 1)
final_outliers = df[outlier_mask & ~true_anomaly_mask]

# 输出结果
print("检测出的异常点数量：", len(final_outliers))
print("异常点索引：", final_outliers.index.tolist())