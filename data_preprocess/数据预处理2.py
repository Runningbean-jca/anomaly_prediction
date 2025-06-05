import pandas as pd
import numpy as np
import os

def process_tube_dataframe(df, ema_alpha=0.3, window=5):
    df = df.copy()

    # 1. 添加 mode_id 并删除 f3, f4, f5
    df['mode_id'] = df['f3'] * 1 + df['f4'] * 2 + df['f5'] * 4
    df.drop(columns=['f3', 'f4', 'f5'], inplace=True)

    # 2. 对 f6-f16 计算滑窗特征
    for col in [f'f{i}' for i in range(6, 17)]:
        df[f'{col}_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_std'] = df[col].rolling(window=window, min_periods=1).std()
        df[f'{col}_delta'] = df[col].diff()
        df[f'{col}_ema'] = df[col].ewm(alpha=ema_alpha, adjust=False).mean()

    # 3. 计算功率 power = f7 * f8
    df['power'] = df['f7'] * df['f8']

    # 4. 计算内部功耗 component_b_power = f10 * f14
    df['component_b_power'] = df['f10'] * df['f14']

    # 5. 温度差（f11~f15）
    temp_cols = [f'f{i}' for i in range(11, 16)]
    for i in range(len(temp_cols)):
        for j in range(i + 1, len(temp_cols)):
            col1, col2 = temp_cols[i], temp_cols[j]
            df[f'temp_diff_{col1}_{col2}'] = np.abs(df[col1] - df[col2])

    # 6. 模式交互项
    df['mode_f7'] = df['mode_id'] * df['f7']
    df['mode_f8'] = df['mode_id'] * df['f8']

    return df


def merge_and_process_all():
    # 合并路径
    base_dir = 'processed_data'
    files = [f'tube{i}_no_label.csv' for i in range(5, 6)]
    dfs = []

    for f in files:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f'[Warning] File not found: {path}')

    # 合并数据
    df_all = pd.concat(dfs, ignore_index=True)
    # df_all.to_csv(os.path.join('detail_processed_data/tube_no_label.csv'), index=False)
    # print(f'[Info] 合并完成，保存为 tube_no_label.csv，总样本数: {len(df_all)}')

    # 特征工程
    df_enhanced = process_tube_dataframe(df_all)
    df_enhanced.to_csv(os.path.join('detail_processed_data/tube5_enhanced_no_label.csv'), index=False)
    print(f'[Info] 特征工程完成，保存为 tube5_enhanced_no_label.csv，最终列数: {df_enhanced.shape[1]}')


if __name__ == '__main__':
    merge_and_process_all()