import pandas as pd
import os

# 定义文件夹路径和文件名
data_dir = 'processed_data'
file_list = [
    'tube1_label.csv',
    'tube2_label.csv',
    'tube3_label.csv',
    'tube4_label.csv',
    'tube5_label.csv'
]

# 按顺序加载每个 CSV 文件
df_list = []
for filename in file_list:
    path = os.path.join(data_dir, filename)
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    df_list.append(df)

# 合并所有 DataFrame
merged_df = pd.concat(df_list, ignore_index=True)

# 保存为新文件
output_path = os.path.join(data_dir, 'tube_label.csv')
merged_df.to_csv(output_path, index=False)
print(f"✅ 合并完成，保存至: {output_path}")