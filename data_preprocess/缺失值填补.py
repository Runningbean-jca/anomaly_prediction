import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


file_path = 'processed_data/tube1_label.csv'
df = pd.read_csv(file_path)

missing_before = df.isnull().sum().sum()
print(f"缺失值总数（处理前）: {missing_before}")

# KNNImputer 进行填补
imputer = KNNImputer(n_neighbors=10, weights="uniform")
df_imputed = imputer.fit_transform(df)

df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

missing_after = df_imputed.isnull().sum().sum()
print(f"缺失值总数（处理后）: {missing_after}")

df_imputed.to_csv('processed_data/tube1_label.csv', index=False)
print("已保存填补后的文件：processed_data/tube1_label.csv")
