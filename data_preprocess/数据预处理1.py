import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据
df = pd.read_csv('data/tube5.csv')

# 1. 生成 f19：f17 或 f18 中任意一个不为 0 就标记为异常
df['f19'] = ((df['f17'] != 0) | (df['f18'] != 0)).astype(int)

# # 2. 生成 f0：f1 为秒数，f0 表示是第几天（以 86400 秒为单位）
# df['f0'] = df['f1'] // 86400

# 3. 处理 f5：将 0.75 → 0，0.875 → 1，1 → 2
f5_mapping = {0.75: 0, 0.875: 1, 1.0: 2}
df['f5'] = df['f5'].map(f5_mapping)

# # 4. 计算 f6 ~ f16 的一阶差分（变化率），并添加为新列
# for i in range(6, 17):  # f6 到 f16 共 11 个特征
#     col = f'f{i}'
#     delta_col = f'delta_{col}'
#     df[delta_col] = df[col].diff()  # 当前值 - 上一个值

# # 5. 构造 f7 ~ f16 的滑动窗口统计特征（window=5）：max, mean, std
# window_size = 5
# for i in range(7, 17):  # f7 到 f16 共 10 个特征
#     col = f'f{i}'
#     df[f'{col}_max_w{window_size}'] = df[col].rolling(window=window_size).max()
#     df[f'{col}_mean_w{window_size}'] = df[col].rolling(window=window_size).mean()
#     df[f'{col}_std_w{window_size}'] = df[col].rolling(window=window_size).std()

# ✅ 保存处理后的数据集
os.makedirs('processed_data', exist_ok=True)
df.to_csv('processed_data/tube5.csv', index=False)

print("✅ 数据处理完成，已保存至 processed_data/tube5.csv")



# # ✅ 确认新特征已添加/修改
# print("✅ 数据集列名如下，已添加 f0 和 f19，并映射 f5：")
# print(df.columns)
#
# # 打印 f19 分布情况
# print("\n📊 f19 异常标记分布：")
# print(df['f19'].value_counts())
#
# # 每天异常率分布（f0 为天数）
# daily_anomaly_rate = df.groupby('f0')['f19'].mean()
#
# # 可视化异常率随天数的变化趋势
# plt.figure(figsize=(10, 5))
# plt.plot(daily_anomaly_rate.index, daily_anomaly_rate.values, marker='o')
# plt.title("📈 每天异常发生比例 (f19 vs f0)")
# plt.xlabel("第几天 (f0)")
# plt.ylabel("异常概率 (f19 平均值)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()