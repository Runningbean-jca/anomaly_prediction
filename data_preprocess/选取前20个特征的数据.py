import pandas as pd

# 指定前 20 个特征名（从截图中提取）
top20_features = [
    'f9_ema', 'f9_mean', 'f9', 'f6_ema', 'f8_ema',
    'f12_ema', 'f7_ema', 'temp_diff_f11_f12', 'temp_diff_f12_f14', 'temp_diff_f12_f15',
    'f9_std', 'f12_delta', 'f12_std', 'f10_ema', 'power',
    'f8_std', 'f6_mean', 'f16_std', 'f6_std', 'f10_std'
]

# === Step 1: 读取原始 CSV ===
df = pd.read_csv('detail_processed_data/tube3_enhanced_no_label.csv')

# === Step 2: 保留这 20 个特征列 ===
df_selected = df[top20_features]

# === Step 3: 保存新文件 ===
output_path = 'detail_processed_data/tube3_enhanced_no_label_20_features.csv'
df_selected.to_csv(output_path, index=False)

print(f"✅ 已成功保存前 20 特征至: {output_path}")