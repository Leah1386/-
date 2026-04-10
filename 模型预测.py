import torch
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')
from model_build import DEVICE, STA_LSTM, LSTM, BiLSTM, GRU

# ===================== 配置 =====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 路径
DATA_PATH = r"D:\20260323复现论文\数据集"
# 预测输出新路径（自动创建文件夹）
SAVE_PATH = r"D:\20260323复现论文\实验结果\预测2"
os.makedirs(SAVE_PATH, exist_ok=True)

# 模型权重加载路径
MODEL_WEIGHT_PATH = r"D:\20260323复现论文\实验结果\模型训练"

# 模型配置表
MODEL_LIST = [
    {"name": "STA-LSTM", "class": STA_LSTM, "path": "sta_lstm_model.pth", "color": "#F24236"},
    {"name": "LSTM", "class": LSTM, "path": "lstm_model.pth", "color": "#2E86AB"},
    {"name": "BiLSTM", "class": BiLSTM, "path": "bilstm_model.pth", "color": "#27AE60"},
    {"name": "GRU", "class": GRU, "path": "gru_model.pth", "color": "#E74C3C"},
]

# ===================== 步骤1：加载公共数据 =====================
print("🔹 1. 加载测试数据集...")
X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))
scaler = joblib.load(os.path.join(DATA_PATH, "scaler.pkl"))
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)


# 反归一化函数
def inverse_transform(y):
    dummy = np.zeros((len(y), 3))
    dummy[:, 0] = y
    return scaler.inverse_transform(dummy)[:, 0]


y_true_inv = inverse_transform(y_test)
results = []  # 保存所有模型指标
all_residuals = []  # 新增：保存所有模型残差（用于统一绘图）

# ===================== 步骤2：批量预测所有模型 =====================
print("\n🔹 2. 开始批量预测所有模型...\n")
for model_cfg in MODEL_LIST:
    model_name = model_cfg["name"]
    ModelClass = model_cfg["class"]
    model_path = model_cfg["path"]

    print(f"👉 正在预测：{model_name}")
    # 加载模型
    model = ModelClass().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHT_PATH, model_path), weights_only=True))
    model.eval()

    # 预测
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    y_pred_inv = inverse_transform(y_pred)
    residual = y_true_inv - y_pred_inv

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)
    results.append([model_name, f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}"])
    all_residuals.append({"name": model_name, "res": residual, "color": model_cfg["color"]})

    # 生成每个模型的专属图表
    # 1. 预测对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label='真实地下水位', color='black', linewidth=2.5)
    plt.plot(y_pred_inv, label=f'{model_name} 预测值', color=model_cfg["color"], linewidth=2.5, linestyle='--')
    plt.title(f'{model_name} 地下水位预测结果对比 (R²={r2:.4f})', fontsize=14)
    plt.xlabel('测试样本序列', fontsize=12)
    plt.ylabel('地下水位 (m)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{model_name.lower()}_prediction.png"), dpi=300)
    plt.close()

    # 2. 残差图
    plt.figure(figsize=(10, 4))
    plt.scatter(range(len(residual)), residual, color=model_cfg["color"], alpha=0.7, s=20)
    plt.axhline(y=0, color='red', linewidth=1.5)
    plt.title(f'{model_name} 预测残差分布', fontsize=14)
    plt.xlabel('测试样本序列', fontsize=12)
    plt.ylabel('残差 (m)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f"{model_name.lower()}_residual.png"), dpi=300)
    plt.close()

# ===================== 步骤3：输出论文对比指标表（控制台+图片） =====================
print("\n" + "=" * 80)
print("📋 多模型地下水位预测精度对比表（论文直接复制）")
print("=" * 80)
print(f"{'模型':<10}\t{'RMSE':<10}\t{'MAE':<10}\t{'R²':<10}")
for res in results:
    print(f"{res[0]:<10}\t{res[1]:<10}\t{res[2]:<10}\t{res[3]:<10}")
print("=" * 80)

# -------------------------- 新增：绘制精度对比表图片 --------------------------
print("\n🔹 4. 生成精度对比表图片...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')
# 表格表头+数据
columns = ["模型", "RMSE", "MAE", "R²"]
table_data = [columns] + results
# 绘制表格（浅蓝配色，PPT专用）
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)
# 表头样式
for i in range(4):
    table[(0, i)].set_facecolor('#1f77b4')
    table[(0, i)].set_text_props(weight='bold', color='white')
# 保存图片
plt.title('多模型地下水位预测精度对比表', fontsize=16, pad=20)
plt.savefig(os.path.join(SAVE_PATH, "precision_comparison_table.png"), dpi=300, bbox_inches='tight')
plt.close()

# ===================== 步骤4：绘制统一多模型对比图 =====================
print("\n🔹 5. 生成多模型预测对比图...")
plt.figure(figsize=(14, 6))
plt.plot(y_true_inv, label='真实值', color='black', linewidth=3, zorder=5)

# 绘制所有模型预测曲线
for model_cfg in MODEL_LIST:
    model_name = model_cfg["name"]
    color = model_cfg["color"]
    for res in results:
        if res[0] == model_name:
            r2 = float(res[3])
            break
    model = model_cfg["class"]().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHT_PATH, model_cfg["path"]), weights_only=True))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    y_pred_inv = inverse_transform(y_pred)

    plt.plot(y_pred_inv, label=f'{model_name} (R²={r2:.4f})', color=color, linewidth=2)

plt.title('各模型地下水位预测结果对比', fontsize=16)
plt.xlabel('测试样本序列', fontsize=14)
plt.ylabel('地下水位 (m)', fontsize=14)
plt.legend(fontsize=12, loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "all_model_comparison.png"), dpi=300)
plt.close()

# -------------------------- 新增：绘制所有模型统一残差分布图 --------------------------
print("\n🔹 6. 生成多模型统一残差对比图...")
plt.figure(figsize=(12, 5))
for res in all_residuals:
    plt.scatter(range(len(res["res"])), res["res"], color=res["color"],
                alpha=0.6, s=18, label=res["name"])
plt.axhline(y=0, color='red', linewidth=2, label='零误差线')
plt.title('多模型预测残差分布对比', fontsize=15)
plt.xlabel('测试样本序列', fontsize=12)
plt.ylabel('残差 (m)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "all_model_residual.png"), dpi=300)
plt.close()

# ===================== 完成 =====================
print(f"\n🎉 所有模型预测完成！")
print(f"📁 结果保存至：{SAVE_PATH}")
print(f"✅ 新增文件：精度对比表图片 + 统一残差对比图")
print(f"✅ 原有文件：4个模型独立图表 + 统一预测对比图")