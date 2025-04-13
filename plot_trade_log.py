import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 读取CSV文件
df = pd.read_csv('trained_agents/trade_log.csv')

# 创建图形和子图
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# 绘制价格变化和交易点
ax1.plot(df['Step'], df['Price'], color='blue', label='Price')
# 标记买入点
buy_points = df[df['Action'] == 1]
ax1.scatter(buy_points['Step'], buy_points['Price'], color='red', marker='^', label='Buy', s=100)
# 标记卖出点
sell_points = df[df['Action'] == -1]
ax1.scatter(sell_points['Step'], sell_points['Price'], color='green', marker='v', label='Sell', s=100)
ax1.set_title('BTC Price Over Time (with Trade Points)')
ax1.set_xlabel('Step')
ax1.set_ylabel('Price')
ax1.grid(True)
ax1.legend()

# 绘制总资产变化和交易点
ax2.plot(df['Step'], df['Total'], color='green', label='Total Assets')
# 标记买入点
ax2.scatter(buy_points['Step'], buy_points['Total'], color='red', marker='^', label='Buy', s=100)
# 标记卖出点
ax2.scatter(sell_points['Step'], sell_points['Total'], color='black', marker='v', label='Sell', s=100)
ax2.set_title('Total Assets Over Time (with Trade Points)')
ax2.set_xlabel('Step')
ax2.set_ylabel('Total Assets')
ax2.grid(True)
ax2.legend()

# 计算相关性和拟合度
# 标准化数据以便比较
price_normalized = (df['Price'] - df['Price'].mean()) / df['Price'].std()
total_normalized = (df['Total'] - df['Total'].mean()) / df['Total'].std()

# 计算皮尔逊相关系数
correlation = np.corrcoef(price_normalized, total_normalized)[0, 1]

# 计算R方值（拟合优度）
slope, intercept, r_value, p_value, std_err = stats.linregress(price_normalized, total_normalized)
r_squared = r_value ** 2

# 绘制拟合度对比图
ax3.scatter(price_normalized, total_normalized, color='gray', alpha=0.5, label='Data Points')
ax3.plot(price_normalized, slope * price_normalized + intercept, color='red', label='Fitted Line')
ax3.set_title(f'Price vs Total Assets (Normalized)\nCorrelation: {correlation:.4f}, R²: {r_squared:.4f}')
ax3.set_xlabel('Normalized Price')
ax3.set_ylabel('Normalized Total Assets')
ax3.grid(True)
ax3.legend()

# 调整子图间距
plt.tight_layout()

# 保存图片
plt.savefig('trained_agents/trade_analysis.png')
plt.close()

# 打印详细的统计信息
print(f"统计分析结果：")
print(f"皮尔逊相关系数: {correlation:.4f}")
print(f"R方值（拟合优度）: {r_squared:.4f}")
print(f"斜率: {slope:.4f}")
print(f"截距: {intercept:.4f}")
print(f"P值: {p_value:.4f}")