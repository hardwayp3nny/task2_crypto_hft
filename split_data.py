import pandas as pd
import os

# 读取CSV文件
df = pd.read_csv(r'f:\finrl\FinRL_Contest_2025\Task_2_FinRL_AlphaSeek_Crypto\data\BTC_1sec.csv')

# 计算分割点
total_rows = len(df)
split_point = int(total_rows * 2/3)

# 分割数据
train_data = df[:split_point]
test_data = df[split_point:]

# 获取表头（第一行）
header = df.iloc[0]

# 保存训练数据
train_data.to_csv(r'f:\finrl\FinRL_Contest_2025\Task_2_FinRL_AlphaSeek_Crypto\data\BTC_1sec_train.csv', index=False)

# 保存测试数据（先添加表头行）
test_data_with_header = pd.concat([pd.DataFrame([header]), test_data])
test_data_with_header.to_csv(r'f:\finrl\FinRL_Contest_2025\Task_2_FinRL_AlphaSeek_Crypto\data\BTC_1sec_test.csv', index=False)

print(f"原始数据总行数: {total_rows}")
print(f"训练集行数: {len(train_data)}")
print(f"测试集行数: {len(test_data)}")