
"""
程序名称：Delhi 小时级 AQI 预测建模数据集构建程序

英文名称：Build Hourly AQI Modeling Dataset

程序用途：
将已经完成空气质量与天气整合的数据，
转换成适用于下一小时 AQI 预测的机器学习数据集。

主要工作：
1. 读取空气质量与天气合并数据；
2. 构建历史 AQI 滞后特征；
3. 构建历史污染物滞后特征；
4. 构建降雨及其他气象特征；
5. 构建小时、星期、月份和季节特征；
6. 设置下一小时 AQI 为预测目标；
7. 检查时间连续性，避免错误的预测时间匹配；
8. 输出建模数据集、缺失率报告和特征说明。

输入：
data/processed/delhi_hourly_air_quality_weather.csv

主要输出：
data/processed/delhi_hourly_modeling_dataset.csv

辅助输出：
outputs/modeling_preparation/

重要说明：
1. 所有预测特征只能来自当前时刻或过去时刻；
2. 不使用下一小时的真实污染物浓度和天气观测值；
3. 保留原始 AQI 大于 500 的记录，不进行截断；
4. 缺失特征暂时保留，由后续模型训练阶段处理；
5. 本程序只构建数据，不训练机器学习模型。

所属阶段：机器学习建模前的数据准备
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ==================================================
# 第一部分：项目路径配置
# ==================================================

# 当前文件位于 code/06_modeling_preparation/
# parents[2] 指向 Air_Quality_Project 根目录

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 已合并的空气质量与天气数据
PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "delhi_hourly_air_quality_weather.csv"
)

# 建模数据输出路径
OUTPUT_FILE = (
    PROCESSED_DIR
    / "delhi_hourly_modeling_dataset.csv"
)

# 建模准备阶段的辅助报告
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "modeling_preparation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 第二部分：读取与检查数据
# ==================================================

print("=" * 65)
print("DELHI HOURLY AQI MODELING DATASET PREPARATION")
print("=" * 65)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到输入文件：{INPUT_FILE}"
    )

print("\n正在读取合并后的空气质量与天气数据...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

# 清理字段名称前后的空格
df.columns = [
    str(col).strip()
    for col in df.columns
]

# 兼容旧版和新版天气字段名称
weather_aliases = {
    "Temperature": "Temperature_C",
    "Humidity": "Humidity_pct",
    "WindSpeed": "WindSpeed_kmh"
}

for old_name, new_name in weather_aliases.items():

    if (
        old_name in df.columns
        and new_name not in df.columns
    ):

        df = df.rename(
            columns={
                old_name: new_name
            }
        )

# 检查必要的数据字段
required_columns = [
    "StationId",
    "Datetime",
    "AQI",
    "Rain_mm",
    "Temperature_C",
    "Humidity_pct",
    "WindSpeed_kmh"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"原始数据缺少必要字段：{missing_columns}"
    )

# 统一站点编号格式
df["StationId"] = (
    df["StationId"]
    .astype(str)
    .str.strip()
)

# 转换时间格式
df["Datetime"] = pd.to_datetime(
    df["Datetime"],
    errors="coerce"
)

# 转换数值字段
numeric_columns = [
    "AQI",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Rain_mm",
    "Temperature_C",
    "Humidity_pct",
    "WindSpeed_kmh"
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# 检查时间与监测站编号
if df["Datetime"].isna().any():

    raise ValueError(
        "Datetime 中存在无效时间记录。"
    )

if df["StationId"].isna().any():

    raise ValueError(
        "StationId 中存在缺失值。"
    )

if df["AQI"].notna().sum() == 0:

    raise ValueError(
        "AQI 字段没有有效数值，无法构建预测数据集。"
    )

# 检查重复的站点小时记录
duplicate_count = int(
    df.duplicated(
        subset=["StationId", "Datetime"]
    ).sum()
)

if duplicate_count > 0:

    raise ValueError(
        f"发现 {duplicate_count} 条重复的站点小时记录，"
        "请先处理重复数据。"
    )

# 按监测站和时间排序
df = df.sort_values(
    ["StationId", "Datetime"]
).reset_index(drop=True)

print("\n原始数据读取完成。")

print("原始记录数：", len(df))

print(
    "监测站数量：",
    df["StationId"].nunique()
)

print(
    "原始 AQI 缺失数量：",
    df["AQI"].isna().sum()
)


# ==================================================
# 第三部分：保留原始 AQI 和极端污染标记
# ==================================================

# 不对原始 AQI 进行截断或删除
df["AQI_Original"] = df["AQI"].copy()

# 当前时刻的极端 AQI 标记
df["Extreme_AQI_Flag_Current"] = (
    df["AQI"] > 500
).astype(int)


# ==================================================
# 第四部分：构建历史 AQI 滞后特征
# ==================================================

print("\n正在构建历史 AQI 特征...")

# 分别按照监测站进行分组
grouped = df.groupby(
    "StationId",
    sort=False
)

# 需要构建的历史时间跨度
aqi_lags = [1, 2, 3, 24]

for lag in aqi_lags:

    # 获取当前站点在前 lag 行的 AQI
    lag_values = grouped["AQI"].shift(lag)

    # 获取对应的历史时间
    lag_times = grouped["Datetime"].shift(lag)

    # 检查当前时间与历史记录是否确实相差 lag 小时
    time_difference = (
        df["Datetime"] - lag_times
    )

    valid_time = (
        time_difference
        == pd.Timedelta(hours=lag)
    )

    # 如果时间不连续，则不使用该滞后值
    df[f"AQI_Lag_{lag}"] = lag_values.where(
        valid_time
    )

# 当前时刻 AQI，作为下一小时预测的输入
df["AQI_Current"] = df["AQI"]


# ==================================================
# 第五部分：构建污染物历史特征
# ==================================================

print("\n正在构建污染物滞后特征...")

pollutant_lag_mapping = {
    "PM2.5": "PM25_Lag_1",
    "PM10": "PM10_Lag_1",
    "NO2": "NO2_Lag_1",
    "CO": "CO_Lag_1",
    "O3": "O3_Lag_1"
}

for source_column, feature_name in pollutant_lag_mapping.items():

    if source_column not in df.columns:

        print(
            f"警告：缺少污染物字段 {source_column}，"
            "跳过该特征。"
        )

        df[feature_name] = np.nan

        continue

    # 获取上一小时污染物浓度
    lag_values = grouped[source_column].shift(1)

    # 检查上一条记录是否确实是一个小时前
    lag_times = grouped["Datetime"].shift(1)

    valid_time = (
        df["Datetime"] - lag_times
        == pd.Timedelta(hours=1)
    )

    df[feature_name] = lag_values.where(
        valid_time
    )


# ==================================================
# 第六部分：构建天气特征
# ==================================================

print("\n正在构建天气特征...")

# 当前时刻已获得的天气信息
df["Rain_mm_Current"] = df["Rain_mm"]

df["Temperature_C_Current"] = (
    df["Temperature_C"]
)

df["Humidity_pct_Current"] = (
    df["Humidity_pct"]
)

df["WindSpeed_kmh_Current"] = (
    df["WindSpeed_kmh"]
)

# 上一小时降雨量
rain_lag_1 = grouped["Rain_mm"].shift(1)

rain_lag_1_time = grouped["Datetime"].shift(1)

rain_lag_1_valid = (
    df["Datetime"] - rain_lag_1_time
    == pd.Timedelta(hours=1)
)

df["Rain_mm_Lag_1"] = rain_lag_1.where(
    rain_lag_1_valid
)

# 前两小时降雨量
rain_lag_2 = grouped["Rain_mm"].shift(2)

rain_lag_2_time = grouped["Datetime"].shift(2)

rain_lag_2_valid = (
    df["Datetime"] - rain_lag_2_time
    == pd.Timedelta(hours=2)
)

rain_lag_2 = rain_lag_2.where(
    rain_lag_2_valid
)

# 计算包括当前小时在内的三小时累计降雨量
# 只有三个连续小时的降雨数据都完整时才计算
df["Rain_3h_Total"] = (
    df["Rain_mm_Current"]
    + df["Rain_mm_Lag_1"]
    + rain_lag_2
)


# ==================================================
# 第七部分：构建时间特征
# ==================================================

print("\n正在构建时间特征...")

# 小时特征
df["Hour"] = df["Datetime"].dt.hour

# 星期特征，Monday = 0
df["DayOfWeek"] = df["Datetime"].dt.dayofweek

# 月份特征
df["Month"] = df["Datetime"].dt.month

# 季节分类
def classify_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Pre-monsoon"

    elif month in [6, 7, 8, 9]:
        return "Monsoon"

    else:
        return "Post-monsoon"


df["Season"] = df["Month"].apply(
    classify_season
)

# 周期性编码
# 避免模型将 23 点和 0 点理解为相距很远
df["Hour_Sin"] = np.sin(
    2 * np.pi * df["Hour"] / 24
)

df["Hour_Cos"] = np.cos(
    2 * np.pi * df["Hour"] / 24
)

# 月份周期编码
df["Month_Sin"] = np.sin(
    2 * np.pi * (df["Month"] - 1) / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * (df["Month"] - 1) / 12
)


# ==================================================
# 第八部分：构建下一小时 AQI 预测目标
# ==================================================

print("\n正在构建下一小时预测目标...")

# 获取同一个监测站下一条记录的时间和 AQI
grouped = df.groupby(
    "StationId",
    sort=False
)

next_datetime = grouped["Datetime"].shift(-1)

next_aqi = grouped["AQI"].shift(-1)

# 只有下一条记录确实是当前时间之后一个小时，
# 才将其作为下一小时 AQI 预测目标
valid_next_hour = (
    next_datetime
    == df["Datetime"] + pd.Timedelta(hours=1)
)

df["Target_Datetime"] = next_datetime.where(
    valid_next_hour
)

df["AQI_Next_Hour"] = next_aqi.where(
    valid_next_hour
)

# 下一小时是否属于极端 AQI 记录
df["AQI_Next_Hour_Extreme_Flag"] = (
    df["AQI_Next_Hour"] > 500
).astype(int)


# 只保留当前 AQI 和下一小时预测目标均有效的记录
# 这样每条样本都具备当前 AQI 基准值

model_df = df.dropna(
    subset=[
        "AQI_Current",
        "AQI_Next_Hour"
    ]
).copy()

# 确认所有建模样本的当前 AQI 都有效
assert model_df["AQI_Current"].notna().all()

# 确认所有预测目标都有效
assert model_df["AQI_Next_Hour"].notna().all()

# 确认每条样本的预测时间都严格相隔一个小时
assert (
    model_df["Target_Datetime"] - model_df["Datetime"]
    == pd.Timedelta(hours=1)
).all()

# 恢复时间顺序
model_df = model_df.sort_values(
    ["StationId", "Datetime"]
).reset_index(drop=True)


# ==================================================
# 第九部分：确定模型输入特征和预测目标
# ==================================================

# 下面的字段是候选预测特征
feature_columns = [
    "StationId",

    # 当前 AQI 和历史 AQI
    "AQI_Current",
    "AQI_Lag_1",
    "AQI_Lag_2",
    "AQI_Lag_3",
    "AQI_Lag_24",

    # 历史污染物
    "PM25_Lag_1",
    "PM10_Lag_1",
    "NO2_Lag_1",
    "CO_Lag_1",
    "O3_Lag_1",

    # 天气特征
    "Rain_mm_Current",
    "Rain_mm_Lag_1",
    "Rain_3h_Total",
    "Temperature_C_Current",
    "Humidity_pct_Current",
    "WindSpeed_kmh_Current",

    # 时间特征
    "Hour",
    "DayOfWeek",
    "Month",
    "Season",
    "Hour_Sin",
    "Hour_Cos",
    "Month_Sin",
    "Month_Cos"
]

# 删除原始数据中不存在的候选特征
feature_columns = [
    col for col in feature_columns
    if col in model_df.columns
]

# 预测目标
target_column = "AQI_Next_Hour"

# 结果元数据字段
metadata_columns = [
    "Datetime",
    "Target_Datetime",
    "AQI_Next_Hour_Extreme_Flag"
]

# 最终输出字段
output_columns = list(
    dict.fromkeys(
        metadata_columns
        + feature_columns
        + [target_column]
    )
)

model_df = model_df[output_columns].copy()


# ==================================================
# 第十部分：统计特征缺失率
# ==================================================

print("\n正在计算建模特征缺失率...")

missingness = pd.DataFrame({

    "Feature": feature_columns,

    "Missing_Count": [
        model_df[col].isna().sum()
        for col in feature_columns
    ],

    "Missing_Percentage": [
        round(
            model_df[col].isna().mean() * 100,
            3
        )
        for col in feature_columns
    ],

    "Data_Type": [
        str(model_df[col].dtype)
        for col in feature_columns
    ]
})

missingness = missingness.sort_values(
    "Missing_Percentage",
    ascending=False
)

missingness.to_csv(
    OUTPUT_DIR / "modeling_feature_missingness.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 第十一部分：生成特征说明表
# ==================================================

feature_descriptions = {

    "StationId": "Monitoring station ID（监测站编号）",

    "AQI_Current": "Current AQI（当前小时 AQI）",

    "AQI_Lag_1": "AQI one hour ago（前一小时 AQI）",

    "AQI_Lag_2": "AQI two hours ago（前两小时 AQI）",

    "AQI_Lag_3": "AQI three hours ago（前三小时 AQI）",

    "AQI_Lag_24": "AQI 24 hours ago（前一天同一时刻 AQI）",

    "PM25_Lag_1": "PM2.5 concentration one hour ago（前一小时 PM2.5）",

    "PM10_Lag_1": "PM10 concentration one hour ago（前一小时 PM10）",

    "NO2_Lag_1": "NO2 concentration one hour ago（前一小时 NO2）",

    "CO_Lag_1": "CO concentration one hour ago（前一小时 CO）",

    "O3_Lag_1": "O3 concentration one hour ago（前一小时 O3）",

    "Rain_mm_Current": "Current available rainfall（当前时刻已获得的降雨量）",

    "Rain_mm_Lag_1": "Rainfall one hour ago（前一小时降雨量）",

    "Rain_3h_Total": "Total rainfall over current and previous two hours（最近三小时累计降雨量）",

    "Temperature_C_Current": "Current temperature（当前气温）",

    "Humidity_pct_Current": "Current relative humidity（当前相对湿度）",

    "WindSpeed_kmh_Current": "Current wind speed（当前风速）",

    "Hour": "Hour of day（一天中的小时）",

    "DayOfWeek": "Day of week（星期）",

    "Month": "Calendar month（月份）",

    "Season": "Season category（季节分类）",

    "Hour_Sin": "Cyclical hour sine encoding（小时周期正弦编码）",

    "Hour_Cos": "Cyclical hour cosine encoding（小时周期余弦编码）",

    "Month_Sin": "Cyclical month sine encoding（月份周期正弦编码）",

    "Month_Cos": "Cyclical month cosine encoding（月份周期余弦编码）",

    "Datetime": "Prediction issue time（预测起点时间）",

    "Target_Datetime": "Target observation time（预测目标对应时间）",

    "AQI_Next_Hour": "Next-hour AQI prediction target（下一小时 AQI 预测目标）",

    "AQI_Next_Hour_Extreme_Flag": "Next-hour extreme AQI flag（下一小时极端 AQI 标记）"
}

feature_dictionary = pd.DataFrame([
    {
        "Feature": col,
        "Description": feature_descriptions.get(
            col,
            "Additional feature（其他特征）"
        ),
        "Role": (
            "Prediction target（预测目标）"
            if col == target_column
            else
            "Metadata（时间或站点信息）"
            if col in metadata_columns
            else
            "Input feature（模型输入特征）"
        )
    }
    for col in output_columns
])

feature_dictionary.to_csv(
    OUTPUT_DIR / "modeling_feature_dictionary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 第十二部分：生成建模数据集汇总
# ==================================================

station_summary = (
    model_df.groupby("StationId")
    .agg(
        Modeling_Records=("AQI_Next_Hour", "count"),
        Target_AQI_Mean=("AQI_Next_Hour", "mean"),
        Target_AQI_Median=("AQI_Next_Hour", "median"),
        Target_AQI_Max=("AQI_Next_Hour", "max"),
        Extreme_AQI_Count=(
            "AQI_Next_Hour_Extreme_Flag",
            "sum"
        )
    )
    .reset_index()
)

station_summary["Extreme_AQI_Percentage"] = (
    station_summary["Extreme_AQI_Count"]
    / station_summary["Modeling_Records"]
    * 100
).round(3)

station_summary.to_csv(
    OUTPUT_DIR / "station_modeling_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# 统计整个建模数据集
summary = pd.DataFrame({

    "Metric": [
        "Original_Rows",
        "Modeling_Rows",
        "Removed_Rows",
        "Station_Count",
        "Feature_Count",
        "Target_Name",
        "Target_Missing_Count",
        "Target_Extreme_AQI_Count",
        "Target_Extreme_AQI_Percentage",
        "Start_Time",
        "End_Time"
    ],

    "Value": [
        len(df),
        len(model_df),
        len(df) - len(model_df),
        model_df["StationId"].nunique(),
        len(feature_columns),
        target_column,
        model_df[target_column].isna().sum(),
        model_df["AQI_Next_Hour_Extreme_Flag"].sum(),
        round(
            model_df["AQI_Next_Hour_Extreme_Flag"].mean() * 100,
            3
        ),
        model_df["Datetime"].min(),
        model_df["Datetime"].max()
    ]
})

summary.to_csv(
    OUTPUT_DIR / "modeling_dataset_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 第十三部分：保存最终建模数据集
# ==================================================

model_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 第十四部分：输出检查报告
# ==================================================

print("\n" + "=" * 65)
print("MODELING DATASET PREPARATION COMPLETED")
print("=" * 65)

print("\n原始数据记录数：", len(df))

print("最终建模记录数：", len(model_df))

print(
    "被排除的记录数：",
    len(df) - len(model_df)
)

print(
    "监测站数量：",
    model_df["StationId"].nunique()
)

print(
    "模型输入特征数量：",
    len(feature_columns)
)

print(
    "预测目标：",
    target_column
)

print(
    "预测时间范围：",
    model_df["Datetime"].min(),
    "至",
    model_df["Target_Datetime"].max()
)

print("\n特征缺失率：")

print(
    missingness.to_string(index=False)
)

print("\n各站点建模记录：")

print(
    station_summary.to_string(index=False)
)

print("\n最终建模数据集保存位置：")

print(OUTPUT_FILE)

print("\n辅助报告保存位置：")

print(OUTPUT_DIR)

print("\n建模数据集构建完成！")
