import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# 读取权重分析报告
file_path = r"f:\finrl\FinRL_Contest_2025\Task_2_FinRL_AlphaSeek_Crypto\analysis\weight_analysis_report.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()

# 提取特征重要性数据
feature_importance = []
for line in lines:
    if line.startswith('Feature '):
        parts = line.strip().split(': ')
        feature_num = int(parts[0].split(' ')[1])
        importance = float(parts[1])
        feature_importance.append((feature_num, importance))

# 转换为DataFrame
df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
df = df.sort_values('Importance', ascending=False)

# 只选择前20个最重要的特征进行可视化
top_n = 20
top_features = df.head(top_n)

# 设置Seaborn风格
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))

# 创建水平条形图
ax = sns.barplot(x='Importance', y='Feature', data=top_features, 
                palette='viridis', orient='h')

# 添加标题和标签
plt.title('Top 20 Features by Importance', fontsize=16, fontweight='bold')
plt.xlabel('Normalized Importance', fontsize=14)
plt.ylabel('Feature ID', fontsize=14)

# 添加数值标签
for i, v in enumerate(top_features['Importance']):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

# 调整布局
plt.tight_layout()

# 保存图像
output_dir = r"f:\finrl\FinRL_Contest_2025\Task_2_FinRL_AlphaSeek_Crypto\analysis"
output_path = os.path.join(output_dir, "feature_importance_top20.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# 创建热力图版本 - 适合论文中展示
plt.figure(figsize=(12, 10))

# 创建热力图数据
heatmap_data = top_features.copy()
heatmap_data['Feature'] = heatmap_data['Feature'].astype(str)
heatmap_matrix = pd.DataFrame({
    'Feature': heatmap_data['Feature'],
    'Importance': heatmap_data['Importance']
}).set_index('Feature')

# 绘制热力图
sns.heatmap(heatmap_matrix.T, annot=True, cmap='YlGnBu', fmt='.3f', 
            linewidths=.5, cbar_kws={'label': 'Importance'})

plt.title('Feature Importance Heatmap (Top 20)', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存热力图
heatmap_path = os.path.join(output_dir, "feature_importance_heatmap.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')

# 创建圆形图 - 更适合论文展示
plt.figure(figsize=(12, 12))

# 准备数据
sizes = top_features['Importance'].values
labels = [f'Feature {int(x)}' for x in top_features['Feature'].values]
colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))

# 绘制饼图
wedges, texts, autotexts = plt.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'edgecolor': 'w', 'linewidth': 1},
    textprops={'fontsize': 12}
)

# 设置自动文本的样式
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_weight('bold')

# 添加标题
plt.title('Feature Importance Distribution (Top 20)', fontsize=16, fontweight='bold')
plt.axis('equal')  # 确保饼图是圆形的

# 保存饼图
pie_path = os.path.join(output_dir, "feature_importance_pie.png")
plt.savefig(pie_path, dpi=300, bbox_inches='tight')

print(f"已生成三种可视化图表：")
print(f"1. 条形图: {output_path}")
print(f"2. 热力图: {heatmap_path}")
print(f"3. 饼图: {pie_path}")
print("请选择最适合您论文的图表格式。")