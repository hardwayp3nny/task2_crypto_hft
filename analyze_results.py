import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

def analyze_trading_results(save_path="trained_agents"):
    """
    分析交易评估结果
    
    参数:
        save_path: 保存结果的路径前缀，与task2_eval.py中使用的相同
    """
    # 设置中文字体支持
    try:
        # 尝试使用微软雅黑字体
        font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc", size=10)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        # 如果找不到微软雅黑，使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        font = FontProperties(size=10)
    
    # 加载保存的数据
    positions = np.load(f"{save_path}_positions.npy")
    net_assets = np.load(f"{save_path}_net_assets.npy")
    btc_positions = np.load(f"{save_path}_btc_positions.npy")
    correct_predictions = np.load(f"{save_path}_correct_predictions.npy")
    
    # 创建时间序列索引
    time_steps = np.arange(len(net_assets))
    
    # 计算收益率
    returns = np.diff(net_assets) / net_assets[:-1]
    
    # 计算累积收益率
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用更现代的风格
    
    # 创建一个包含多个子图的图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 资产价值变化
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(time_steps, net_assets, 'b-', linewidth=1.5)
    ax1.set_title('净资产价值变化', fontproperties=font)
    ax1.set_xlabel('时间步', fontproperties=font)
    ax1.set_ylabel('净资产', fontproperties=font)
    ax1.ticklabel_format(style='plain', axis='y')  # 避免科学计数法
    ax1.grid(True, alpha=0.3)
    
    # 2. 累积收益率
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time_steps[1:], cumulative_returns * 100, 'g-', linewidth=1.5)
    ax2.set_title('累积收益率 (%)', fontproperties=font)
    ax2.set_xlabel('时间步', fontproperties=font)
    ax2.set_ylabel('累积收益率 (%)', fontproperties=font)
    ax2.grid(True, alpha=0.3)
    
    # 3. 持仓变化
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(time_steps, btc_positions, 'r-', linewidth=1.5)
    ax3.set_title('BTC持仓量变化', fontproperties=font)
    ax3.set_xlabel('时间步', fontproperties=font)
    ax3.set_ylabel('BTC持仓量', fontproperties=font)
    ax3.grid(True, alpha=0.3)
    
    # 4. 预测正确率分析
    win_count = np.sum(correct_predictions == 1)
    loss_count = np.sum(correct_predictions == -1)
    neutral_count = np.sum(correct_predictions == 0)
    total_trades = len(correct_predictions) - neutral_count
    
    if total_trades > 0:
        win_rate = win_count / total_trades * 100
    else:
        win_rate = 0
    
    labels = ['正确预测', '错误预测', '无交易']
    sizes = [win_count, loss_count, neutral_count]
    colors = ['#66b3ff', '#ff9999', '#99ff99']
    explode = (0.1, 0, 0)
    
    ax4 = fig.add_subplot(3, 2, 4)
    wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=None, colors=colors, 
                                      autopct='%1.1f%%', shadow=True, startangle=90)
    # 单独设置图例，避免中文显示问题
    ax4.legend(wedges, labels, loc="center right", bbox_to_anchor=(1.1, 0.5), fontsize=9, prop=font)
    ax4.axis('equal')
    ax4.set_title(f'预测结果分布 (胜率: {win_rate:.2f}%)', fontproperties=font)
    
    # 5. 收益率分布
    ax5 = fig.add_subplot(3, 2, 5)
    sns.histplot(returns, kde=True, ax=ax5, color='blue', alpha=0.6)
    ax5.set_title('收益率分布', fontproperties=font)
    ax5.set_xlabel('收益率', fontproperties=font)
    ax5.set_ylabel('频率', fontproperties=font)
    
    # 添加收益率统计信息
    mean_return = np.mean(returns) * 100
    std_return = np.std(returns) * 100
    skew = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    stats_text = (f"平均收益率: {mean_return:.4f}%\n"
                 f"收益率标准差: {std_return:.4f}%\n"
                 f"偏度: {skew:.4f}\n"
                 f"峰度: {kurtosis:.4f}")
    
    ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontproperties=font)
    
    # 6. 交易行为分析
    # 修复交易行为统计方法
    # 不应该使用position_changes，而应该使用action_ints或correct_predictions来统计
    
    # 方法1：根据correct_predictions来推断交易行为
    # 正确预测和错误预测都表示有交易发生
    buys = 0
    sells = 0
    holds = 0
    
    # 遍历correct_predictions，统计不同类型的交易
    for i in range(len(correct_predictions)):
        if correct_predictions[i] == 1 or correct_predictions[i] == -1:
            # 判断是买入还是卖出
            # 这里我们需要根据positions的变化来判断
            if i > 0 and positions[i] > positions[i-1]:
                buys += 1
            elif i > 0 and positions[i] < positions[i-1]:
                sells += 1
            else:
                # 尝试执行交易但未成功（可能是资金不足或其他原因）
                holds += 1
        else:
            # correct_predictions为0表示没有交易
            holds += 1
    
    # 检查数据是否有效，避免饼图绘制错误
    if buys + sells + holds > 0:  # 确保有数据可以绘制
        labels = ['买入', '卖出', '持有']
        sizes = [buys, sells, holds]
        # 过滤掉数量为0的类别
        filtered_labels = []
        filtered_sizes = []
        filtered_colors = []
        colors = ['#66b3ff', '#ff9999', '#99ff99']
        
        for i, size in enumerate(sizes):
            if size > 0:
                filtered_labels.append(labels[i])
                filtered_sizes.append(size)
                filtered_colors.append(colors[i])
        
        ax6 = fig.add_subplot(3, 2, 6)
        if filtered_sizes:  # 确保过滤后还有数据
            wedges, texts, autotexts = ax6.pie(filtered_sizes, labels=None, colors=filtered_colors, 
                                             autopct='%1.1f%%', shadow=True, startangle=90)
            # 单独设置图例，避免中文显示问题
            ax6.legend(wedges, filtered_labels, loc="center right", bbox_to_anchor=(1.1, 0.5), fontsize=9, prop=font)
            ax6.axis('equal')
            ax6.set_title('交易行为分布', fontproperties=font)
        else:
            ax6.text(0.5, 0.5, '没有足够的交易数据', 
                    horizontalalignment='center', verticalalignment='center',
                    fontproperties=font)
            ax6.set_title('交易行为分布 (无数据)', fontproperties=font)
    else:
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.text(0.5, 0.5, '没有交易数据', 
                horizontalalignment='center', verticalalignment='center',
                fontproperties=font)
        ax6.set_title('交易行为分布 (无数据)', fontproperties=font)
    
    # 添加更多统计信息到图表标题
    max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns) * 100
    
    # 修改夏普比率计算方法，导入与task2_eval.py相同的函数
    try:
        # 尝试导入相同的metrics模块
        from metrics import sharpe_ratio as metrics_sharpe_ratio
        sharpe = metrics_sharpe_ratio(returns)
    except:
        # 如果导入失败，使用原来的计算方法
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # 假设252个交易日/年
        print("警告: 无法导入metrics模块中的sharpe_ratio函数，使用替代计算方法")
    
    plt.suptitle(f'交易策略评估结果分析\n最大回撤: {max_drawdown:.2f}%, 夏普比率: {sharpe:.4f}',
                fontsize=16, fontproperties=font)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_path}_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印详细统计信息
    print("\n===== 交易策略评估结果分析 =====")
    print(f"总交易次数: {total_trades}")
    print(f"正确预测次数: {win_count}")
    print(f"错误预测次数: {loss_count}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {mean_return:.4f}%")
    print(f"收益率标准差: {std_return:.4f}%")
    print(f"夏普比率: {sharpe:.4f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"偏度: {skew:.4f}")
    print(f"峰度: {kurtosis:.4f}")
    print(f"买入次数: {buys}")
    print(f"卖出次数: {sells}")
    print(f"持有次数: {holds}")
    print(f"初始资产: {net_assets[0]:.2f}")
    print(f"最终资产: {net_assets[-1]:.2f}")
    print(f"总收益率: {(net_assets[-1]/net_assets[0] - 1) * 100:.2f}%")
    
    # 保存统计结果到CSV文件
    stats_df = pd.DataFrame({
        '指标': ['总交易次数', '正确预测次数', '错误预测次数', '胜率(%)', 
                '平均收益率(%)', '收益率标准差(%)', '夏普比率', '最大回撤(%)',
                '偏度', '峰度', '买入次数', '卖出次数', '持有次数',
                '初始资产', '最终资产', '总收益率(%)'],
        '值': [total_trades, win_count, loss_count, win_rate,
              mean_return, std_return, sharpe, max_drawdown,
              skew, kurtosis, buys, sells, holds,
              net_assets[0], net_assets[-1], (net_assets[-1]/net_assets[0] - 1) * 100]
    })
    
    stats_df.to_csv(f"{save_path}_stats.csv", index=False)
    print(f"\n统计结果已保存到 {save_path}_stats.csv")
    print(f"可视化结果已保存到 {save_path}_analysis.png")
    
    return stats_df

if __name__ == "__main__":
    # 使用与task2_eval.py相同的save_path
    save_path = "trained_agents"
    analyze_trading_results(save_path)