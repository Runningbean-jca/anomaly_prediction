import pandas as pd
import os

# 读取处理后的原始数据（已包含 f0, f19, f6-f16）
df = pd.read_csv('processed_data/tube5.csv')

# 特征范围
numeric_features = [f'f{i}' for i in range(6, 17)]  # f6~f16

# 需要排除的列（不参与聚合/输出）
exclude_cols = ['f1', 'f2', 'f17', 'f18']

# 聚合函数
agg_dict = {}

# f6 ~ f16 做 mean, max, min, std
for col in numeric_features:
    agg_dict[col + '_mean'] = (col, 'mean')
    agg_dict[col + '_max'] = (col, 'max')
    agg_dict[col + '_min'] = (col, 'min')
    agg_dict[col + '_std'] = (col, 'std')

# f19 异常率
agg_dict['Anomaly_rate'] = ('f19', 'mean')

# 统计每一天的记录数
agg_dict['times'] = ('f0', 'count')

# 按 f0 分组并聚合
df_grouped = df.groupby('f0').agg(**agg_dict).reset_index()

# 将 Anomaly_rate ≠ 0 的赋值为 1，==0 的保持为 0
df_grouped['anomaly'] = df_grouped['Anomaly_rate'].apply(lambda x: 1 if x != 0 else 0)

# 删除原 Anomaly_rate 列（可选）
df_grouped.drop(columns=['Anomaly_rate'], inplace=True)

# 计算 f6_mean ~ f16_mean 的变化率（差分）
for col in numeric_features:
    mean_col = col + '_mean'
    diff_col = mean_col + '_diff'
    df_grouped[diff_col] = df_grouped[mean_col].diff()

# 构建目标列顺序
ordered_cols = ['f0', 'times', 'anomaly']
for col in numeric_features:
    ordered_cols += [
        f'{col}_mean',
        f'{col}_mean_diff',
        f'{col}_max',
        f'{col}_min',
        f'{col}_std',
    ]

# 应用新列顺序
df_grouped = df_grouped[ordered_cols]

# 保存到新目录
os.makedirs('day_data', exist_ok=True)
df_grouped.to_csv('day_data/tube5_day.csv', index=False)

print("✅ 按天聚合处理完成，结果已保存至 day_data/tube5_day.csv")