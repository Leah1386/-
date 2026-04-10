import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# -------------------------- 1. 配置路径（自动适配你的文件） --------------------------
# 输入：预处理完成的数据
input_file = r"D:\20260323复现论文\数据\preprocessed_data.csv"
# 输出：模型训练用的数据集
output_dir = r"D:\20260323复现论文\数据集"
os.makedirs(output_dir, exist_ok=True)  # 自动创建文件夹

# 时序预测参数（水文标准：用前7天预测第8天）
TIME_STEP = 7
# 特征列（和预处理后一致）
FEATURE_COLS = ["ground_water", "temperature", "rainfall"]
# 预测目标（只预测地下水位）
TARGET_COL = "ground_water"

# -------------------------- 2. 加载平稳数据 --------------------------
df = pd.read_csv(input_file, parse_dates=["Date"], index_col="Date")
data = df[FEATURE_COLS].values
print(f"📊 预处理后数据总量：{len(data)} 行，{data.shape[1]} 个特征")

# -------------------------- 3. 数据归一化（LSTM必须！） --------------------------
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # 缩放到 0-1 之间
data_scaled = scaler.fit_transform(data)
print("✅ 数据归一化完成（0-1区间）")

# -------------------------- 4. 滑动窗口构建时序序列（核心！） --------------------------
def create_sequences(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        # 前 time_step 天数据 → 输入 X
        X.append(data[i-time_step : i, :])
        # 第 i 天的地下水位 → 标签 y
        y.append(data[i, 0])  # 0 代表 ground_water 列
    return np.array(X), np.array(y)

# 构建训练样本
X, y = create_sequences(data_scaled, TIME_STEP)
print(f"🔗 序列构建完成：")
print(f"   输入 X 形状：{X.shape} → [样本数, 时间步, 特征数]")
print(f"   标签 y 形状：{y.shape} → [样本数, 预测目标]")

# -------------------------- 5. 划分训练集 / 测试集（8:2 标准比例） --------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\n📂 数据集划分：")
print(f"   训练集：{len(X_train)} 条")
print(f"   测试集：{len(X_test)} 条")

# -------------------------- 6. 保存数据集（直接用于模型训练） --------------------------
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)
# 保存归一化器（后续反算真实水位用）
import joblib
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

print(f"\n🎉 数据构建全部完成！")
print(f"📁 所有训练文件保存至：{output_dir}")

# -------------------------- 7. 可视化样本（可选，论文用图） --------------------------
plt.figure(figsize=(10, 4))
plt.plot(df[TARGET_COL], label="预处理后地下水位", color="#2E86AB")
plt.axvline(x=df.index[train_size+TIME_STEP], color="red", linestyle="--", label="训练/测试分割线")
plt.title("地下水位时序数据（训练集+测试集）", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "train_test_split.png"), dpi=300)
plt.close()
print(f"🖼️ 数据集分割图已保存！")