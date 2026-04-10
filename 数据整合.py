import pandas as pd

# -------------------------- 1. 配置文件路径（替换为你的实际文件名） --------------------------
groundwater_file = r"D:\20260323复现论文\数据\Naqugroundwater.csv"   # 地下水位数据
meteorology_file = r"D:\20260323复现论文\数据\Naqumeteorology_clean.csv"    # 气象数据
output_file = r"D:\20260323复现论文\数据\combined_core_data.csv"       # 只保留核心字段的输出文件

# -------------------------- 2. 读取并处理地下水位数据 --------------------------
gw_df = pd.read_csv(groundwater_file)
# 解析中文日期："2019年1月1日" → 标准datetime
gw_df["Date"] = pd.to_datetime(gw_df["Date"], format="%Y年%m月%d日")
# 只保留日期+地下水位（Well-1）
gw_df = gw_df[["Date", "Well-1"]].rename(columns={"Well-1": "ground_water"})
print("✅ 地下水位数据处理完成，共{}条记录".format(len(gw_df)))

# -------------------------- 3. 读取并筛选气象数据（只保留核心字段） --------------------------
met_df = pd.read_csv(meteorology_file)
# 解析斜杠日期："2018/1/1" → 标准datetime
met_df["Data"] = pd.to_datetime(met_df["Data"], format="%Y-%m-%d")
# 只保留核心字段：日期+气温+降雨（可选加湿度/风速，取消下面注释即可）
met_df_core = met_df.rename(columns={
    "Data": "Date",
    "Daily average temperature": "temperature",
    "Daily precipitation": "rainfall"
    # "Relative air humidity": "humidity",       # 可选保留
    # "Daily average wind speed": "wind_speed"  # 可选保留
})
# 只保留核心列
met_df_core = met_df_core[["Date", "temperature", "rainfall"]]  # 若加可选字段，这里补充

# 筛选2019-2021年数据（和地下水位对齐）
start_date = pd.to_datetime("2019-01-01")
end_date = pd.to_datetime("2021-12-31")
met_df_filtered = met_df_core[(met_df_core["Date"] >= start_date) & (met_df_core["Date"] <= end_date)]
print("✅ 气象数据筛选完成（仅核心字段），共{}条记录".format(len(met_df_filtered)))

# -------------------------- 4. 按日期拼接（只保留共同日期） --------------------------
combined_df = pd.merge(gw_df, met_df_filtered, on="Date", how="inner")
print("🔗 合并完成，最终核心数据集共{}条记录".format(len(combined_df)))

# -------------------------- 5. 检查缺失值并保存 --------------------------
print("\n⚠️ 缺失值统计（核心字段）：")
print(combined_df.isnull().sum())
# 填充少量缺失值（线性插值，和论文预处理逻辑一致）
combined_df = combined_df.interpolate(method="linear").dropna()

# 保存最终核心数据集
combined_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n🎉 核心数据已保存为 {output_file}，可直接用于data_preprocess.py！")