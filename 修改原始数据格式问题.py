import pandas as pd
import os

# 1. 读取CSV文件（正确路径）
file_path = r"D:\20260323复现论文\数据\Naqumeteorology.csv"
met_df = pd.read_csv(
    file_path,
    encoding="utf-8",
    header=0  # 第一行是列名
)

print("=== 原始数据预览 ===")
print(met_df.head(3))

# 2. 数据清洗核心步骤
## 步骤1：删除单位行（索引0行，内容是℃/W/㎡等）
met_df = met_df.drop(index=0).reset_index(drop=True)
print("\n=== 删除单位行后 ===")
print(met_df.head(2))

## 步骤2：清理Data列的脏数据（仅保留有效日期）
# 过滤掉非日期格式的行（比如(yyyy-mm-dd)，这里已经删了，做双重保障）
met_df = met_df[met_df["Data"] != "(yyyy-mm-dd)"]

## 步骤3：转换日期列（格式匹配2018/1/1）
try:
    # %Y/%m/%d 完美匹配2018/1/1格式（自动兼容无前置零的月/日）
    met_df["Data"] = pd.to_datetime(met_df["Data"], format="%Y/%m/%d")
    print("\n✅ 日期列转换成功！")
except Exception as e:
    # 容错方案：自动识别格式
    met_df["Data"] = pd.to_datetime(met_df["Data"], format="mixed", errors="coerce")
    print(f"\n✅ 日期列转换成功（自动识别）：{e}")

## 步骤4：处理数值列（转成数值类型，方便后续分析）
# 定义需要转换的数值列（排除Data列）
numeric_cols = [col for col in met_df.columns if col != "Data"]
for col in numeric_cols:
    # 去除空格+转数值（无法转换的设为NaN）
    met_df[col] = pd.to_numeric(met_df[col].str.strip(), errors="coerce")

# 3. 最终结果验证
print("\n=== 最终清洗后的数据 ===")
print(met_df.head(3))
print(f"\n✅ 数据总行数：{len(met_df)}")
print(f"✅ 日期列数据类型：{met_df['Data'].dtype}")
print(f"✅ 温度列数据类型：{met_df['Daily average temperature'].dtype}")

# 4. 可选：保存清洗后的数据（方便后续使用）
output_path = r"D:\20260323复现论文\数据\Naqumeteorology_clean.csv"
met_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\n✅ 清洗后的数据已保存至：{output_path}")