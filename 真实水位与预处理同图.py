import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- 路径配置（和你的文件完全一致） --------------------------
ORIGINAL_DATA = r"D:\20260323复现论文\数据\combined_core_data.csv"       # 原始真实数据
PRE_DATA = r"D:\20260323复现论文\数据\preprocessed_data.csv"             # 预处理后数据
SAVE_PLOT = r"D:\20260323复现论文\数据\真实vs预处理水位对比图.png"        # 输出图片

# -------------------------- 加载数据 --------------------------
# 真实地下水位
df_real = pd.read_csv(ORIGINAL_DATA, parse_dates=["Date"], index_col="Date")
# 预处理后（差分平稳）水位
df_pre = pd.read_csv(PRE_DATA, parse_dates=["Date"], index_col="Date")

# -------------------------- 绘图设置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文显示
plt.rcParams['axes.unicode_minus'] = False        # 负号显示
plt.figure(figsize=(14, 5))

# -------------------------- 核心：双轴绘图（同图展示两条曲线） --------------------------
# 左Y轴：真实地下水位（浅蓝色）
ax1 = plt.gca()
line1 = ax1.plot(df_real["ground_water"], color="#1f77b4", linewidth=2.5, label="真实地下水位")
ax1.set_xlabel("日期", fontsize=12)
ax1.set_ylabel("真实地下水位 (m)", fontsize=12, color="#1f77b4")
ax1.tick_params(axis='y', labelcolor="#1f77b4")
ax1.grid(alpha=0.3)

# 右Y轴：预处理后差分水位（橙色）
ax2 = ax1.twinx()
line2 = ax2.plot(df_pre["ground_water"], color="#ff7f0e", linewidth=2.5, label="预处理后水位(差分)")
ax2.set_ylabel("预处理后水位 (差分平稳序列)", fontsize=12, color="#ff7f0e")
ax2.tick_params(axis='y', labelcolor="#ff7f0e")

# 合并图例 + 标题
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best", fontsize=12)
plt.title("真实地下水位与预处理后水位序列对比", fontsize=15, pad=15)

# 保存高清图
plt.tight_layout()
plt.savefig(SAVE_PLOT, dpi=300)
plt.close()

print("🎉 绘图完成！图片已保存至：")
print(SAVE_PLOT)