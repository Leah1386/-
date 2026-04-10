import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 新增：划分验证集
import warnings

warnings.filterwarnings('ignore')
# =========================================
#🔥🔥🔥模型导入（切入对比模型，STA_LSTM、LSTM、BiLSTM、GRU）
from model_build import DEVICE, BiLSTM

# ===================== 配置 =====================
INPUT_SIZE = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1
#数据量比较小，所以轮数最好在100-150
EPOCHS = 150
#将学习率降低，避免模型训练太快，容易震荡，导致注意力机制学偏。
LR = 0.0005

# 路径
DATA_PATH = r"D:\20260323复现论文\数据集"
SAVE_PATH = r"D:\20260323复现论文\实验结果\模型训练"
#下面这个代码，代表生成文件夹
os.makedirs(SAVE_PATH, exist_ok=True)

# 绘图中文设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 步骤1：加载数据集 + 划分验证集 =====================
print("🔹 1. 加载数据集并划分验证集...")
X_train = np.load(os.path.join(DATA_PATH, "X_train.npy"))
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_train.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
scaler = joblib.load(os.path.join(DATA_PATH, "scaler.pkl"))

#  从训练集中划分 20% 作为验证集（用于监控过拟合）
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# ===================== 步骤2：初始化模型 =====================
print("🔹 2. 初始化模型...")
# 🔥🔥🔥模型初始化（STA_LSTM、LSTM、BiLSTM、GRU）
model = BiLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===================== 步骤3：训练 + 验证损失计算 =====================
print("🔹 3. 开始训练（含验证集）...")
train_losses = []  # 训练损失
val_losses = []  # 验证损失（新增）

for epoch in range(EPOCHS):
    # -------- 训练阶段 --------
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -------- 验证阶段（不更新参数） --------
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    # 保存损失
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    # 打印日志
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | 训练损失: {loss.item():.6f} | 验证损失: {val_loss.item():.6f}")

# 🔥🔥🔥保存模型（STA_LSTM、LSTM、BiLSTM、GRU）
torch.save(model.state_dict(), os.path.join(SAVE_PATH, "bilstm_model.pth"))

# ===================== 步骤4：绘制 训练/验证 双损失曲线 =====================
print("🔹 4. 生成训练&验证损失对比图...")
plt.figure(figsize=(10, 4))
plt.plot(train_losses, color='#27AE60', linewidth=2, label='训练损失 (Train Loss)')
plt.plot(val_losses, color='#E74C3C', linewidth=2, label='验证损失 (Val Loss)')
#🔥🔥🔥 保存图名（STA_LSTM、LSTM、BiLSTM、GRU）
plt.title('BiLSTM 训练与验证损失曲线', fontsize=14)
plt.xlabel('训练轮数 (Epoch)', fontsize=12)
plt.ylabel('损失值 (Loss)', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
#🔥🔥🔥 保存高清对比图（STA_LSTM、LSTM、BiLSTM、GRU）
plt.savefig(os.path.join(SAVE_PATH, "BiLSTM-train_val_loss_curve.png"), dpi=300)
plt.close()

print("\n🎉 训练完成！双损失曲线已保存！")