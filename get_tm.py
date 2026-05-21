# ==================== 导入必要库 ====================
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import callback
import warnings
warnings.filterwarnings('ignore')  # 忽略警告，让输出更干净

print("[OK] 库导入成功，开始构建预测模型...")

# ==================== 1. 数据准备 ====================
# 读取实际数据
file_path = r'C:\Users\Administrator\Desktop\DLtset.csv'
df = pd.read_csv(file_path, parse_dates=['date'])

df = df.dropna(axis=1, how='all')  # 删除空列

# 删除result列为空的行（避免滞后特征产生大量空值）
df = df.dropna(subset=['result'])
df.reset_index(drop=True, inplace=True)

# 列名映射（确保列名一致）
print(f"[INFO] 原始数据列: {list(df.columns)}")
print(f"[INFO] 删除空值后数据形状: {df.shape}")

# ==================== 2. 时间特征工程 ====================
# 自动判断节假日和周末
df['dayofweek'] = df['date'].dt.dayofweek  # 0=周一, 6=周日
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear

# 自动判断中国法定节假日（2026年）
def is_chinese_holiday(date):
    """判断是否为2026年中国法定节假日"""
    month = date.month
    day = date.day

    # 元旦 2026-01-01 至 2026-01-03
    if month == 1 and day in [1, 2, 3]:
        return True
    # 春节 2026-02-17 至 2026-02-23
    if month == 2 and day in range(17, 24):
        return True
    # 清明节 2026-04-04 至 2026-04-06
    if month == 4 and day in [4, 5, 6]:
        return True
    # 劳动节 2026-05-01 至 2026-05-05
    if month == 5 and day in range(1, 6):
        return True
    # 端午节 2026-06-19 至 2026-06-21
    if month == 6 and day in [19, 20, 21]:
        return True
    # 中秋节 2026-09-25
    if month == 9 and day == 25:
        return True
    # 国庆节 2026-10-01 至 2026-10-07
    if month == 10 and day in range(1, 8):
        return True

    return False

df['is_holiday'] = df['date'].apply(is_chinese_holiday).astype(int)

# 工作日特征（非周末且非节假日）
df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)

# 季度特征
df['quarter'] = df['date'].dt.quarter

# ==================== 2.1 节假日类型特征（针对电力负荷优化） ====================
def get_holiday_type(date):
    """
    根据日期返回节假日类型（影响电力负荷的铁路运载程度）
    - 0: 普通工作日
    - 1: 普通周末
    - 2: 短期假期(3天): 元旦、清明、端午、中秋
    - 3: 长期假期(5天): 五一、国庆
    - 4: 春节(7天): 影响最大
    """
    month, day = date.month, date.day

    # 元旦(3天): 1.1-1.3
    if month == 1 and day in [1, 2, 3]:
        return 2
    # 春节(7天): 2.17-2.23 - 铁路运载最大
    if month == 2 and day in range(17, 24):
        return 4
    # 清明(3天): 4.4-4.6
    if month == 4 and day in [4, 5, 6]:
        return 2
    # 五一(5天): 5.1-5.5 - 铁路运载明显
    if month == 5 and day in range(1, 6):
        return 3
    # 端午(3天): 6.19-6.21
    if month == 6 and day in [19, 20, 21]:
        return 2
    # 中秋(1天): 9.25
    if month == 9 and day == 25:
        return 2
    # 国庆(7天): 10.1-10.7 - 铁路运载最大
    if month == 10 and day in range(1, 8):
        return 4

    # 判断周末或工作日
    dayofweek = date.dayofweek
    if dayofweek >= 5:  # 周末
        return 1
    else:  # 工作日
        return 0

df['holiday_type'] = df['date'].apply(get_holiday_type)

print(f"[INFO] 数据形状: {df.shape}")
print(f"   周末天数: {df['is_weekend'].sum()}")
print(f"   节假日天数: {df['is_holiday'].sum()}")
print(f"   工作日天数: {df['is_workday'].sum()}")
print(f"   节假日类型分布:\n{df['holiday_type'].value_counts().sort_index()}")
print(df[['date', 'is_weekend', 'is_holiday', 'is_workday', 'holiday_type']].head(10))

# ==================== 3. 天气特征增强 ====================
# 温差
df['temp_diff'] = df['temp_max'] - df['temp_min']
# 温度偏离舒适区（20℃）的程度
df['temp_deviation'] = np.abs(df['temp_avg'] - 20)

# ==================== 3.1 温度区间特征（针对电力负荷优化） ====================
def get_temperature_zone(temp):
    """
    根据温度返回电力负荷影响区间
    - 0: 低温区(<15℃): 采暖负荷明显
    - 1: 舒适区(15-25℃): 空调负荷低
    - 2: 高温区(>25℃): 制冷负荷明显
    """
    if temp < 15:
        return 0  # 低温区
    elif temp <= 25:
        return 1  # 舒适区
    else:
        return 2  # 高温区

df['temp_zone'] = df['temp_avg'].apply(get_temperature_zone)

# 温度区间统计特征
df['is_low_temp'] = (df['temp_zone'] == 0).astype(int)  # 低温区标识
df['is_comfort_zone'] = (df['temp_zone'] == 1).astype(int)  # 舒适区标识
df['is_high_temp'] = (df['temp_zone'] == 2).astype(int)  # 高温区标识

# 低温程度（温度越低，采暖负荷越高）
df['heating_load'] = np.maximum(0, 15 - df['temp_avg'])  # 低于15℃的部分

# 高温程度（温度越高，制冷负荷越高）
df['cooling_load'] = np.maximum(0, df['temp_avg'] - 25)  # 高于25℃的部分

print(f"[INFO] 温度区间分布:")
print(f"   低温区(<15℃): {df['is_low_temp'].sum()} 天")
print(f"   舒适区(15-25℃): {df['is_comfort_zone'].sum()} 天")
print(f"   高温区(>25℃): {df['is_high_temp'].sum()} 天")

# 交叉特征
df['holiday_temp'] = df['is_holiday'] * df['temp_avg']
df['weekend_temp'] = df['is_weekend'] * df['temp_avg']

# 节假日类型与温度的交叉特征
df['holiday_type_temp'] = df['holiday_type'] * df['temp_avg']

# 温度区间与节假日的交叉特征
df['high_temp_holiday'] = df['is_high_temp'] * df['is_holiday']  # 高温+节假日（制冷+铁路）
df['low_temp_holiday'] = df['is_low_temp'] * df['is_holiday']    # 低温+节假日（采暖+铁路）

# ==================== 3.2 铁路运载相关特征 ====================
# 长期假期标识（铁路运载明显增加）
df['is_long_holiday'] = ((df['holiday_type'] == 3) | (df['holiday_type'] == 4)).astype(int)

# 春节标识（铁路运载最大）
df['is_spring_festival'] = (df['holiday_type'] == 4).astype(int)

# 长期假期与温度的交叉特征
df['long_holiday_temp'] = df['is_long_holiday'] * df['temp_avg']
df['long_holiday_high_temp'] = df['is_long_holiday'] * df['is_high_temp']  # 长期假期+高温
df['long_holiday_low_temp'] = df['is_long_holiday'] * df['is_low_temp']    # 长期假期+低温

# ==================== 3.3 工作日负荷特征 ====================
# 工作日与温度的交叉特征
df['workday_temp'] = df['is_workday'] * df['temp_avg']
df['workday_high_temp'] = df['is_workday'] * df['is_high_temp']  # 工作日+高温
df['workday_low_temp'] = df['is_workday'] * df['is_low_temp']    # 工作日+低温

# 降雨对铁路电力负荷的影响
df['rain_impact'] = df['rainfall'] * df['is_workday']  # 工作日雨天影响更大

# 天气综合影响评分
df['weather_load_factor'] = df['heating_load'] + df['cooling_load'] + df['rainfall'] * 0.5

# 假期前/后特征（负荷可能异常）
# 这里简单使用周末作为假期前后的近似标识
df['is_before_weekend'] = (df['dayofweek'] == 4).astype(int)  # 周五
df['is_after_weekend'] = (df['dayofweek'] == 0).astype(int)  # 周一

print(f"[INFO] 铁路运载相关特征:")
print(f"   长期假期天数: {df['is_long_holiday'].sum()} 天")
print(f"   春节天数: {df['is_spring_festival'].sum()} 天")

# ==================== 4. 全部数据用于训练（无测试集） ====================
# 合并用于特征工程（滞后特征需要历史数据）
combined = df.copy()

# 对result列创建滞后特征
for lag in [1, 2, 3, 7]:
    combined[f'result_lag_{lag}d'] = combined['result'].shift(lag)

# 滚动统计特征
combined['result_roll_mean_7d'] = combined['result'].shift(1).rolling(window=7).mean()
combined['result_roll_std_7d'] = combined['result'].shift(1).rolling(window=7).std()

# 删除因滞后产生的空值（前7行）
combined.dropna(inplace=True)
combined.reset_index(drop=True, inplace=True)

print(f"[INFO] 全量数据: {len(combined)}天（全部用于训练）")

# 准备模型输入 - 只选择数值类型的列
feature_cols = [c for c in combined.columns if c not in ['date', 'result', 'Unnamed: 10'] and pd.api.types.is_numeric_dtype(combined[c])]
X_train, y_train = combined[feature_cols], combined['result'].astype(float)

print(f"[INFO] 特征数量: {len(feature_cols)}")
print(f"[INFO] 使用的特征列: {feature_cols}")

# ==================== 5. 训练 LightGBM 模型（全部数据） ====================
print("[INFO] 开始训练 LightGBM 模型（全部数据）...")

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
}

train_data = lgb.Dataset(X_train, label=y_train)

# 全部数据训练，不设测试集
model = lgb.train(
    params,
    train_data,
    num_boost_round=500  # 使用固定轮数，不过早停止
)

print(f"[OK] 训练完成！")

# ==================== 6. 训练集评估 ====================
y_pred = model.predict(X_train, num_iteration=model.num_trees())

mae = mean_absolute_error(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
mask = y_train > 1
mape = np.mean(np.abs((y_train[mask] - y_pred[mask]) / y_train[mask])) * 100

print("\n" + "="*50)
print(f"[RESULT] 训练集评估结果（{len(y_train)}天）:")
print(f"   MAE  : {mae:.2f}")
print(f"   RMSE : {rmse:.2f}")
print(f"   MAPE : {mape:.2f}%")
print("="*50)

# ==================== 7. 输出结果 ====================
results_df = pd.DataFrame({
    '日期': combined['date'].values,
    '实际值': y_train.values,
    '训练预测值': y_pred,
})
results_df.to_csv(r'C:\Users\Administrator\PyCharmMiscProject\cache\training_results.csv', index=False, encoding='utf-8-sig')
print("\n[OK] 训练结果已保存为 'cache/training_results.csv'")

# ==================== 8. 特征重要性分析 ====================
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\n[INFO] Top 10 重要特征:")
print(importance_df.head(10).to_string(index=False))

# ==================== 9. 保存模型 ====================
model.save_model('load_prediction_DLmodel.txt')
print(f"\n[OK] 模型已保存为 'load_prediction_DLmodel.txt'")

print("\n[OK] 完成！")
