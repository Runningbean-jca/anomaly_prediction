import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
file_path = 'day_data/tube1_day_no_label.csv'
df = pd.read_csv(file_path)

# 2. 查看数据缺失情况
print("🔍 缺失值统计：")
print(df.isnull().sum())

# # 3. 查看每列特征的数据类型
# print("\n🔍 特征数据类型：")
# print(df.dtypes)
#
# # 4. 检查是否存在完全重复的行
# print("\n🔍 完全重复的行数：")
# print(df.duplicated().sum())
#
# # 5. 检查异常值（使用 IQR 方法）
# def detect_outliers_iqr(column):
#     Q1 = column.quantile(0.25)
#     Q3 = column.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     outliers = column[(column < lower_bound) | (column > upper_bound)]
#     return len(outliers), lower_bound, upper_bound
#
# print("\n🔍 异常值检测（IQR 方法）：")
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# for col in numeric_cols:
#     outlier_count, lb, ub = detect_outliers_iqr(df[col])
#     print(f"{col}: {outlier_count} 个异常值 （下界={lb:.3f}, 上界={ub:.3f}）")
#
# # （可选）箱型图可视化异常值分布
# plt.figure(figsize=(16, 6))
# sns.boxplot(data=df[numeric_cols], orient="h")
# plt.title("📦 各数值特征的箱型图（异常值可视化）")
# plt.tight_layout()
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取数据
# df = pd.read_csv('data/tube5.csv')
#
# # 设置横坐标为 f2（采样次数）
# x = df['f2']
#
# # 要绘制的特征列（f6 到 f16）
# feature_cols = [f'f{i}' for i in range(6, 17)]
#
# # 设置画布：一行三列，共 11 个子图
# plt.figure(figsize=(18, 14))
# for idx, col in enumerate(feature_cols, 1):
#     plt.subplot(4, 3, idx)  # 4行3列，第idx个图（最多12个图）
#     plt.plot(x, df[col], label=col, color='tab:blue')
#     plt.xlabel('f2 (Sample Count)')
#     plt.ylabel(col)
#     plt.title(f'{col} vs f2')
#     plt.grid(True)
#     plt.tight_layout()
#
# # 去掉第12张图的空白（因为只有11张图）
# plt.subplots_adjust(hspace=0.4, wspace=0.3)
# plt.suptitle('📈 f6 ~ f16 特征随时间(f2)的变化趋势', fontsize=16, y=1.02)
# plt.show()