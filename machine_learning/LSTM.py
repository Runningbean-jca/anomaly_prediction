import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ------------------------------
# LSTM 模型定义
# ------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        out = torch.sigmoid(self.fc(last_hidden))
        return out.squeeze()

# ------------------------------
# 1. 数据加载与滑窗构造
# ------------------------------
file_paths = sorted(glob.glob('day_data/tube1_day.csv'))
df_all = pd.concat([pd.read_csv(p) for p in file_paths], ignore_index=True).sort_values(by='f0').reset_index(drop=True)

top20_features = [
    'f9_max', 'f9_mean', 'f9_std', 'f9_min', 'f9_mean_diff',
    'f8_mean', 'f10_mean', 'f16_mean', 'f16_max', 'f6_std',
    'f10_min', 'f12_min', 'f10_mean_diff', 'f15_std', 'f6_mean_diff',
    'f8_std', 'f8_max', 'f10_max', 'f15_max', 'f7_mean_diff'
]
window_size = 3
X, y = [], []

for i in range(len(df_all) - window_size):
    window = df_all.iloc[i:i + window_size][top20_features].values
    target = df_all.iloc[i + window_size]['anomaly']
    X.append(window)
    y.append(target)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# ------------------------------
# 2. 数据集划分
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

X_train_flat = X_train.reshape(len(X_train), -1)
df_train = pd.DataFrame(X_train_flat)
df_train['anomaly'] = y_train
df_0 = df_train[df_train['anomaly'] == 0]
df_1 = df_train[df_train['anomaly'] == 1]

df_0_down = resample(df_0, replace=False, n_samples=50, random_state=42)
df_train_bal = pd.concat([df_0_down, df_1])
X_train_bal = df_train_bal.drop(columns='anomaly').values.reshape(-1, window_size, len(top20_features))
y_train_bal = df_train_bal['anomaly'].values

smote = SMOTE(sampling_strategy={1: 80}, random_state=42)
X_resampled_flat, y_resampled = smote.fit_resample(X_train_bal.reshape(len(X_train_bal), -1), y_train_bal)
X_resampled = X_resampled_flat.reshape(-1, window_size, len(top20_features))

# ------------------------------
# 3. 模型训练
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim=len(top20_features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

X_tensor = torch.tensor(X_resampled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_resampled, dtype=torch.float32).to(device)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/30 - Loss: {loss.item():.4f}")

# ------------------------------
# 4. 模型预测与评估
# ------------------------------
model.eval()
with torch.no_grad():
    y_prob = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve (LSTM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (LSTM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
print(f"📌 最佳 F1 阈值: {best_thresh:.3f}")

y_pred = (y_prob >= best_thresh).astype(int)
print("📊 LSTM 分类评估报告：")
print(classification_report(y_test, y_pred))