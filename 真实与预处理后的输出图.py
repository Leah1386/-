import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- 【唯一配置】路径和你之前完全一致 --------------------------
# 原始数据（直接画真实水位，最稳妥）
ORIGINAL_DATA = r"D:\20260323复现论文\数据\combined_core_data.csv"
# 输出图片保存路径
REAL_WATER_PLOT = r"D:\20260323复现论文\数据\真实地下水位图.png"
PRE_PROCESS_PLOT = r"D:\20260323复现论文\数据\预处理后水位图.png"

# -------------------------- 1. 绘制【真实地下水位图】（核心需求） --------------------------
print("正在绘制 真实地下水位图...")
df_original = pd.read_csv(ORIGINAL_DATA, parse_dates=["Date"], index_col="Date")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 真实水位图
plt.figure(figsize=(12, 4))
plt.plot(df_original["ground_water"], color="#1f77b4", linewidth=2, label="真实地下水位")
plt.title("那曲流域真实地下水位时序变化", fontsize=14)
plt.xlabel("日期")
plt.ylabel("地下水位 (m)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REAL_WATER_PLOT, dpi=300)
plt.close()

# -------------------------- 2. 绘制【预处理后差分水位图】（对比用） --------------------------
print("正在绘制 预处理后水位图...")
# 直接用你之前预处理好的文件（正确文件名）
df_pre = pd.read_csv(r"D:\20260323复现论文\数据\preprocessed_data.csv", parse_dates=["Date"], index_col="Date")

plt.figure(figsize=(12, 4))
plt.plot(df_pre["ground_water"], color="#FF7F0E", linewidth=2, label="预处理后(差分平稳序列)")
plt.title("那曲流域预处理后地下水位序列", fontsize=14)
plt.xlabel("日期")
plt.ylabel("水位变化量 (差分后)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PRE_PROCESS_PLOT, dpi=300)
plt.close()

# -------------------------- 完成 --------------------------
print(f"\n🎉 绘图完成！")
print(f"📊 真实水位图：{REAL_WATER_PLOT}")
print(f"📊 预处理图：{PRE_PROCESS_PLOT}")