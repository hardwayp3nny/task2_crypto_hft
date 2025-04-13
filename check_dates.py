import pandas as pd

# 读取数据
df = pd.read_csv("./data/BTC_1sec.csv")

# 将system_time转换为datetime格式
df['system_time'] = pd.to_datetime(df['system_time'])

# 获取时间范围信息
start_time = df['system_time'].min()
end_time = df['system_time'].max()
total_days = (end_time - start_time).days
total_seconds = (end_time - start_time).total_seconds()

print(f"数据起始时间: {start_time}")
print(f"数据结束时间: {end_time}")
print(f"总计天数: {total_days}天")
print(f"总计小时: {total_seconds/3600:.2f}小时")
print(f"总计数据条数: {len(df)}条")
print(f"平均每秒数据条数: {len(df)/total_seconds:.2f}条")