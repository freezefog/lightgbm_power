# ==================== 使用已训练模型根据天气数据预测 ====================
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("[OK] 库导入成功，加载预测模型...")

# ==================== 1. 加载已训练的模型 ====================
model_path = r'c:\Users\Administrator\PyCharmMiscProject\cache\load_prediction_model.txt'
model = lgb.Booster(model_file=model_path)
print(f"[OK] 模型加载成功！特征: {model.num_feature()} 个")

# ==================== 2. 读取历史数据用于计算滞后特征 ====================
file_path = r'C:\Users\Administrator\Desktop\tset.csv'
df_history = pd.read_csv(file_path, parse_dates=['date'])
df_history = df_history.dropna(subset=['result'])
df_history.reset_index(drop=True, inplace=True)
print(f"[INFO] 历史数据: {len(df_history)} 行 ({df_history['date'].min().date()} 至 {df_history['date'].max().date()})")

# ==================== 3. 读取预测天气数据 (tmp.csv) ====================
weather_file = r'C:\Users\Administrator\Desktop\tmp.csv'  # 用户提供的天气数据文件
try:
    df_weather = pd.read_csv(weather_file, parse_dates=['date'])
    print(f"[OK] 成功读取天气数据: {len(df_weather)} 行")
except FileNotFoundError:
    print(f"[ERROR] 找不到天气数据文件: {weather_file}")
    print("[INFO] 请确保 tmp.csv 文件在当前目录中")
    exit(1)

print(f"[INFO] 预测日期范围: {df_weather['date'].min().date()} 至 {df_weather['date'].max().date()}")

# ==================== 4. 定义节假日和调休判断函数 ====================
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

def is_workday_adjustment(date):
    """
    判断是否为调休工作日（本来是周末但需要上班）
    2026年调休安排：
    - 清明后：4月26日（周日）上班
    - 五一前：4月30日（周四）上班
    - 五一后：5月6日（周三）上班, 5月9日（周六）上班
    - 端午后：6月27日（周六）上班
    - 国庆后：10月11日（周日）上班
    """
    month = date.month
    day = date.day

    # 清明节调休
    if month == 4 and day == 26:  # 4月26日周日上班
        return True
    # 五一节调休
    if month == 4 and day == 30:  # 4月30日周四上班
        return True
    if month == 5 and day == 6:   # 5月6日周三上班
        return True
    if month == 5 and day == 9:   # 5月9日周六上班
        return True
    # 端午节调休
    if month == 6 and day == 27:  # 6月27日周六上班
        return True
    # 国庆节调休
    if month == 10 and day == 11: # 10月11日周日上班
        return True

    return False

# ==================== 5. 构建特征工程 ====================
df_predict = df_weather.copy()

# 时间特征
df_predict['dayofweek'] = df_predict['date'].dt.dayofweek
df_predict['day'] = df_predict['date'].dt.day  # 添加day列
df_predict['is_weekend'] = (df_predict['dayofweek'] >= 5).astype(int)
df_predict['month'] = df_predict['date'].dt.month
df_predict['dayofyear'] = df_predict['date'].dt.dayofyear
df_predict['is_holiday'] = df_predict['date'].apply(is_chinese_holiday).astype(int)
df_predict['is_workday_adjustment'] = df_predict['date'].apply(is_workday_adjustment).astype(int)
# 工作日：非周末 且 非节假日 且 非调休
df_predict['is_workday'] = ((df_predict['is_weekend'] == 0) & (df_predict['is_holiday'] == 0) & (df_predict['is_workday_adjustment'] == 0)).astype(int)
# 调休日视为工作日
df_predict['is_workday'] = df_predict['is_workday'] | df_predict['is_workday_adjustment']
df_predict['quarter'] = df_predict['date'].dt.quarter

# 天气增强特征
df_predict['temp_diff'] = df_predict['temp_max'] - df_predict['temp_min']
df_predict['temp_deviation'] = np.abs(df_predict['temp_avg'] - 20)
df_predict['holiday_temp'] = df_predict['is_holiday'] * df_predict['temp_avg']
df_predict['weekend_temp'] = df_predict['is_weekend'] * df_predict['temp_avg']

# ==================== 5.1 节假日类型特征 ====================
def get_holiday_type_from_row(row):
    """
    根据日期返回节假日类型（影响电力负荷的铁路运载程度）
    - 0: 普通工作日
    - 1: 普通周末
    - 2: 短期假期(3天): 元旦、清明、端午、中秋
    - 3: 长期假期(5天): 五一、国庆
    - 4: 春节(7天): 影响最大
    """
    date = row['date']
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
    if row['is_weekend'] == 1:
        return 1
    else:
        return 0

df_predict['holiday_type'] = df_predict.apply(get_holiday_type_from_row, axis=1)

# ==================== 5.2 温度区间特征 ====================
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

df_predict['temp_zone'] = df_predict['temp_avg'].apply(get_temperature_zone)
df_predict['is_low_temp'] = (df_predict['temp_zone'] == 0).astype(int)
df_predict['is_comfort_zone'] = (df_predict['temp_zone'] == 1).astype(int)
df_predict['is_high_temp'] = (df_predict['temp_zone'] == 2).astype(int)

# 低温程度和高温程度
df_predict['heating_load'] = np.maximum(0, 15 - df_predict['temp_avg'])
df_predict['cooling_load'] = np.maximum(0, df_predict['temp_avg'] - 25)

# 交叉特征
df_predict['holiday_type_temp'] = df_predict['holiday_type'] * df_predict['temp_avg']
df_predict['high_temp_holiday'] = df_predict['is_high_temp'] * df_predict['is_holiday']
df_predict['low_temp_holiday'] = df_predict['is_low_temp'] * df_predict['is_holiday']

# ==================== 5.3 铁路运载相关特征 ====================
df_predict['is_long_holiday'] = ((df_predict['holiday_type'] == 3) | (df_predict['holiday_type'] == 4)).astype(int)
df_predict['is_spring_festival'] = (df_predict['holiday_type'] == 4).astype(int)
df_predict['long_holiday_temp'] = df_predict['is_long_holiday'] * df_predict['temp_avg']
df_predict['long_holiday_high_temp'] = df_predict['is_long_holiday'] * df_predict['is_high_temp']
df_predict['long_holiday_low_temp'] = df_predict['is_long_holiday'] * df_predict['is_low_temp']

# ==================== 5.4 工作日负荷特征 ====================
df_predict['workday_temp'] = df_predict['is_workday'] * df_predict['temp_avg']
df_predict['workday_high_temp'] = df_predict['is_workday'] * df_predict['is_high_temp']
df_predict['workday_low_temp'] = df_predict['is_workday'] * df_predict['is_low_temp']

# 降雨对电力负荷的影响
df_predict['rain_impact'] = df_predict['rainfall'] * df_predict['is_workday']

# 天气综合影响评分
df_predict['weather_load_factor'] = df_predict['heating_load'] + df_predict['cooling_load'] + df_predict['rainfall'] * 0.5

# 假期前/后特征
df_predict['is_before_weekend'] = (df_predict['dayofweek'] == 4).astype(int)
df_predict['is_after_weekend'] = (df_predict['dayofweek'] == 0).astype(int)

# ==================== 5.5 季节性上升期特征 ====================
def is_seasonal_rising_period(date, temp_avg):
    """
    判断是否为季节性上升期（电力负荷恢复/上升阶段）
    - 电力负荷呈现明显的季节性周期：
      1-2月: 年初低位（春节影响）
      3-4月: 逐步回升期
      5-6月: 明显上升期（工业恢复+制冷开始）
      7-8月: 高位期（制冷高峰）
      9-10月: 逐步下降期
      11-12月: 年末高位（采暖开始）
    
    返回上升期强度: 0-1（0=下降期, 0.5=平稳期, 1=强烈上升期）
    """
    month = date.month
    day = date.day
    dayofyear = date.dayofyear
    
    # 计算当年的等效"天数"用于周期性判断
    # 假设1月1日为周期起点
    
    # 1. 五一假期后的恢复上升期 (5月1日-5月10日)
    if month == 5 and day <= 10:
        # 假期刚结束，工作日恢复，负荷明显上升
        if day <= 5:
            return 0.7  # 假期中，可能还处于恢复中
        else:
            return 0.9  # 工作日，强烈上升期
    # 2. 五一前一周的备货期 (4月25日-4月30日)
    elif month == 4 and day >= 25:
        return 0.6  # 备货期，负荷开始上升
    # 3. 夏季高温来临前的积累期 (6月)
    elif month == 6 and temp_avg >= 25:
        return 0.8  # 进入高温区，负荷上升
    # 4. 夏季高峰后的小幅下降期 (9月)
    elif month == 9 and day <= 15:
        return 0.4  # 高峰期过后，小幅下降
    # 5. 采暖期开始 (11月)
    elif month == 11 and day >= 15:
        return 0.7  # 采暖期开始，负荷上升
    # 6. 春节前两周 (1月下旬-2月初)
    elif month == 1 and day >= 20:
        return 0.3  # 春节前，负荷下降
    # 7. 春季温和期 (3-4月)
    elif month in [3, 4]:
        return 0.5  # 平稳过渡期
    # 8. 夏季高峰 (7-8月)
    elif month in [7, 8]:
        return 0.9  # 高温期，负荷高位
    # 9. 秋季温和期 (10月前半月)
    elif month == 10 and day <= 15:
        return 0.5  # 假期高峰后，稳定期
    # 10. 年末高位 (12月)
    elif month == 12:
        return 0.8  # 采暖期，负荷高位
    else:
        return 0.5  # 默认平稳期

df_predict['seasonal_rising'] = df_predict.apply(lambda row: is_seasonal_rising_period(row['date'], row['temp_avg']), axis=1)

# 季节性上升期与其他特征交叉
df_predict['rising_temp'] = df_predict['seasonal_rising'] * df_predict['temp_avg']
df_predict['rising_holiday'] = df_predict['seasonal_rising'] * df_predict['is_holiday']
df_predict['rising_workday'] = df_predict['seasonal_rising'] * df_predict['is_workday']
df_predict['is_post_holiday_rise'] = ((df_predict['month'] == 5) & (df_predict['day'] >= 6) & (df_predict['day'] <= 10) & (df_predict['is_workday'] == 1)).astype(int)

print(f"\n[INFO] 季节性特征:")
print(df_predict[['date', 'seasonal_rising', 'is_post_holiday_rise']].to_string(index=False))

print(f"\n[INFO] 预测日期特征:")
print(df_predict[['date', 'is_weekend', 'is_holiday', 'is_workday', 'holiday_type', 'temp_zone', 'seasonal_rising']].to_string(index=False))

# ==================== 6. 计算日期相似度函数（针对电力负荷预测优化） ====================
def get_temperature_zone(temp):
    """
    根据温度返回电力负荷影响区间
    - 低温区(<15℃): 采暖负荷明显
    - 舒适区(15-25℃): 空调负荷低
    - 高温区(>25℃): 制冷负荷明显
    """
    if temp < 15:
        return 'low_temp'  # 低温区
    elif temp <= 25:
        return 'comfort'   # 舒适区
    else:
        return 'high_temp' # 高温区

def get_holiday_type(row):
    """
    根据日期返回节假日类型（影响电力负荷的铁路运载程度）
    - 0: 普通工作日
    - 1: 普通周末
    - 2: 短期假期(3天): 元旦、清明、端午、中秋
    - 3: 长期假期(5天): 五一、国庆
    - 4: 春节(7天): 影响最大
    """
    if row['is_workday'] == 1:
        return 0  # 普通工作日
    elif row['is_weekend'] == 1 and row['is_holiday'] == 0:
        return 1  # 普通周末
    elif row['is_holiday'] == 1:
        # 判断具体假期类型
        date = row['date']
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

    return 0  # 默认为工作日

def calculate_similarity(row1, row2):
    """
    计算两个日期之间的相似度（针对电力负荷预测优化）

    电力负荷预测特点：
    1. 温度区间影响：电力负荷对温度呈非线性关系，不同区间（低温/舒适/高温）的负荷差异显著
       - 低温区(<15℃): 采暖负荷高
       - 舒适区(15-25℃): 空调负荷低
       - 高温区(>25℃): 制冷负荷高

    2. 节假日铁路运载：长期假期铁路客运量大，显著影响电力负荷
       - 普通工作日：基准负荷
       - 普通周末：负荷略低
       - 短期假期(3天)：铁路增加不明显，负荷影响中等
       - 长期假期(5天+)：铁路运载明显增加，负荷影响显著
       - 春节：铁路运载最大，负荷影响最大

    3. 季节性上升期：某些时期电力负荷呈现系统性上升或下降趋势
       - 五一假期后工作日：负荷明显上升
       - 夏季高温来临：负荷持续上升
       - 冬季采暖期：负荷高位

    权重分配：
    - 温度区间匹配：35%（考虑区间和区间内差异）
    - 节假日类型匹配：30%（重点考虑长期假期影响）
    - 降雨量：10%（雨天可能影响铁路电力负荷）
    - 季节性时期：25%（新增：捕捉周期性上升/下降趋势）
    """
    # === 温度相似度（区间影响）===
    zone1 = get_temperature_zone(row1['temp_avg'])
    zone2 = get_temperature_zone(row2['temp_avg'])

    # 区间匹配权重最高
    if zone1 == zone2:
        zone_match_score = 1.0
        # 同一区间内，计算具体温度差异
        temp_diff = abs(row1['temp_avg'] - row2['temp_avg'])
        # 舒适区(15-25°C)：温度变化对结果影响较小，容忍度高(15度)
        # 高温区/低温区：容忍度低(5度)
        tolerance = 15 if zone1 == 'comfort' else 5
        temp_diff_score = max(0, 1 - temp_diff / tolerance)
        temp_score = 0.5 * zone_match_score + 0.5 * temp_diff_score
    else:
        # 跨区间：相似度较低
        temp_score = 0.3

    # === 节假日类型相似度（铁路运载影响）===
    holiday_type1 = row1['holiday_type']
    holiday_type2 = row2['holiday_type']

    # 完全匹配
    if holiday_type1 == holiday_type2:
        holiday_score = 1.0
    # 类型接近（如短期假期vs工作日）
    elif abs(holiday_type1 - holiday_type2) == 1:
        holiday_score = 0.6
    # 类型差异较大（如长期假期vs工作日）
    else:
        holiday_score = 0.2

    # === 降雨量相似度 ===
    rain_diff = abs(row1['rainfall'] - row2['rainfall'])
    rain_score = max(0, 1 - rain_diff / 20)  # 假设20mm雨量差异为完全不相似

    # === 季节性上升期相似度 ===
    # 考虑季节性时期和五一假期后恢复期的匹配
    seasonal1 = row1.get('seasonal_rising', 0.5)
    seasonal2 = row2.get('seasonal_rising', 0.5)
    seasonal_diff = abs(seasonal1 - seasonal2)
    seasonal_score = 1 - seasonal_diff * 0.5  # 最大差异时分数不低于0.5
    
    # 五一假期后恢复期加成
    post_holiday1 = row1.get('is_post_holiday_rise', 0)
    post_holiday2 = row2.get('is_post_holiday_rise', 0)
    if post_holiday1 == post_holiday2 and post_holiday1 == 1:
        seasonal_score = max(seasonal_score, 0.9)  # 两者都是假期后恢复期，高分
    
    # 月份接近度（考虑季节性）
    month1 = row1['date'].month
    month2 = row2['date'].month
    month_diff = abs(month1 - month2)
    # 考虑跨年情况（如12月和1月）
    if month_diff > 6:
        month_diff = 12 - month_diff
    month_score = 1 - month_diff / 6  # 同月最高分，差6个月最低分
    seasonal_score = 0.6 * seasonal_score + 0.4 * month_score

    # === 时间接近度分数 ===
    days_diff = abs((row1['date'] - row2['date']).days)
    # 每7天降0.1，最低0.4
    time_score = max(0.4, 1 - days_diff / 70)

    # === 综合相似度（加权平均）===
    # 温度和节假日匹配仍然重要，但时间接近度也很关键
    similarity = 0.35 * temp_score + 0.30 * holiday_score + 0.10 * rain_score + 0.10 * seasonal_score + 0.15 * time_score
    return similarity

def calculate_time_score(row1, row2):
    """
    计算时间接近度分数
    越接近，得分越高
    """
    latest_date = max(row1['date'], row2['date']) if isinstance(row1['date'], pd.Timestamp) else max(row1['date'], row2['date'])
    # 计算天数差距
    if isinstance(row1['date'], pd.Timestamp) and isinstance(row2['date'], pd.Timestamp):
        days_diff = abs((row1['date'] - row2['date']).days)
    else:
        days_diff = abs(row1['date'] - row2['date']).days if hasattr(row1['date'], 'days') else 0
    # 每7天降0.1，最低0.5
    time_score = max(0.5, 1 - days_diff / 70)
    return time_score

# ==================== 7. 为历史数据添加特征 ====================
# 为历史数据添加与预测数据相同的特征
df_history['dayofweek'] = df_history['date'].dt.dayofweek
df_history['day'] = df_history['date'].dt.day  # 添加day列
df_history['is_weekend'] = (df_history['dayofweek'] >= 5).astype(int)
df_history['month'] = df_history['date'].dt.month
df_history['dayofyear'] = df_history['date'].dt.dayofyear
df_history['is_holiday'] = df_history['date'].apply(is_chinese_holiday).astype(int)
df_history['is_workday_adjustment'] = df_history['date'].apply(is_workday_adjustment).astype(int)
# 工作日：非周末 且 非节假日 且 非调休
df_history['is_workday'] = ((df_history['is_weekend'] == 0) & (df_history['is_holiday'] == 0) & (df_history['is_workday_adjustment'] == 0)).astype(int)
# 调休日视为工作日
df_history['is_workday'] = df_history['is_workday'] | df_history['is_workday_adjustment']
df_history['quarter'] = df_history['date'].dt.quarter
df_history['temp_diff'] = df_history['temp_max'] - df_history['temp_min']
df_history['temp_deviation'] = np.abs(df_history['temp_avg'] - 20)
df_history['holiday_temp'] = df_history['is_holiday'] * df_history['temp_avg']
df_history['weekend_temp'] = df_history['is_weekend'] * df_history['temp_avg']

# 节假日类型特征
df_history['holiday_type'] = df_history.apply(get_holiday_type_from_row, axis=1)

# 温度区间特征
df_history['temp_zone'] = df_history['temp_avg'].apply(get_temperature_zone)
df_history['is_low_temp'] = (df_history['temp_zone'] == 0).astype(int)
df_history['is_comfort_zone'] = (df_history['temp_zone'] == 1).astype(int)
df_history['is_high_temp'] = (df_history['temp_zone'] == 2).astype(int)
df_history['heating_load'] = np.maximum(0, 15 - df_history['temp_avg'])
df_history['cooling_load'] = np.maximum(0, df_history['temp_avg'] - 25)

# 交叉特征
df_history['holiday_type_temp'] = df_history['holiday_type'] * df_history['temp_avg']
df_history['high_temp_holiday'] = df_history['is_high_temp'] * df_history['is_holiday']
df_history['low_temp_holiday'] = df_history['is_low_temp'] * df_history['is_holiday']

# 铁路运载相关特征
df_history['is_long_holiday'] = ((df_history['holiday_type'] == 3) | (df_history['holiday_type'] == 4)).astype(int)
df_history['is_spring_festival'] = (df_history['holiday_type'] == 4).astype(int)
df_history['long_holiday_temp'] = df_history['is_long_holiday'] * df_history['temp_avg']
df_history['long_holiday_high_temp'] = df_history['is_long_holiday'] * df_history['is_high_temp']
df_history['long_holiday_low_temp'] = df_history['is_long_holiday'] * df_history['is_low_temp']

# 工作日负荷特征
df_history['workday_temp'] = df_history['is_workday'] * df_history['temp_avg']
df_history['workday_high_temp'] = df_history['is_workday'] * df_history['is_high_temp']
df_history['workday_low_temp'] = df_history['is_workday'] * df_history['is_low_temp']

# 降雨影响
df_history['rain_impact'] = df_history['rainfall'] * df_history['is_workday']

# 天气综合影响
df_history['weather_load_factor'] = df_history['heating_load'] + df_history['cooling_load'] + df_history['rainfall'] * 0.5

# 假期前后特征
df_history['is_before_weekend'] = (df_history['dayofweek'] == 4).astype(int)
df_history['is_after_weekend'] = (df_history['dayofweek'] == 0).astype(int)

# 检查历史数据
if len(df_history) == 0:
    print(f"\n[ERROR] 历史数据为空！")
    print(f"[INFO] 请确保历史数据文件包含有效数据")
    exit(1)

print(f"\n[INFO] 历史数据准备完成: {len(df_history)} 条记录")

# ==================== 8. 逐日预测（使用相似日期的滞后特征） ====================
feature_cols = [
    'temp_min', 'temp_avg', 'temp_max', 'rainfall',
    'dayofweek', 'is_weekend', 'month', 'dayofyear',
    'is_holiday', 'is_workday', 'quarter', 'holiday_type',
    'temp_diff', 'temp_deviation', 'temp_zone',
    'is_low_temp', 'is_comfort_zone', 'is_high_temp',
    'heating_load', 'cooling_load',
    'holiday_temp', 'weekend_temp', 'holiday_type_temp',
    'high_temp_holiday', 'low_temp_holiday',
    'is_long_holiday', 'is_spring_festival',
    'long_holiday_temp', 'long_holiday_high_temp', 'long_holiday_low_temp',
    'workday_temp', 'workday_high_temp', 'workday_low_temp',
    'rain_impact', 'weather_load_factor',
    'is_before_weekend', 'is_after_weekend',
    # 滞后特征
    'result_lag_1d', 'result_lag_2d', 'result_lag_3d', 'result_lag_7d',
    'result_roll_mean_7d', 'result_roll_std_7d'
]

predictions = []

# 用于滚动预测：存储已预测的结果
predicted_results = []

# 用于追踪工作日和周末的配对关系
def get_workday_weekend_gap_by_temp(df, target_temp, tolerance=5):
    """
    根据温度范围获取工作日和周末的差值
    舒适区(15-25°C)容忍度更大，高温/低温区容忍度小
    """
    temp_zone = get_temperature_zone(target_temp)
    if temp_zone == 'comfort':
        tolerance = 8  # 舒适区容忍度8度
    else:
        tolerance = 5
    
    # 排除节假日
    same_type = df[(df['is_holiday'] == 0)]

    # 温度接近的工作日
    workdays_temp = same_type[same_type['is_workday'] == 1]['temp_avg'].values
    workdays_result = same_type[same_type['is_workday'] == 1]['result'].values
    
    # 温度接近的周末
    weekends_temp = same_type[(same_type['is_weekend'] == 1) & (same_type['is_workday'] == 0)]['temp_avg'].values
    weekends_result = same_type[(same_type['is_weekend'] == 1) & (same_type['is_workday'] == 0)]['result'].values
    
    # 找温度最接近的配对
    best_gap = 150  # 默认差值
    best_count = 0
    
    for i, wt in enumerate(workdays_temp):
        if abs(wt - target_temp) <= tolerance:
            # 找一个温度接近的周末
            for j, st in enumerate(weekends_temp):
                if abs(st - target_temp) <= tolerance:
                    gap = workdays_result[i] - weekends_result[j]
                    best_gap = (best_gap * best_count + gap) / (best_count + 1)
                    best_count += 1
                    break
    
    return best_gap if best_count > 0 else 150

def find_similar_type_by_temp(df, target_temp, target_is_workday, tolerance=10):
    """
    根据温度和工作类型查找相似的历史日期
    综合考虑温度接近度和时间接近度
    """
    same_type = df[(df['is_holiday'] == 0)]

    if target_is_workday == 1:
        candidates = same_type[same_type['is_workday'] == 1]
    else:
        candidates = same_type[(same_type['is_weekend'] == 1) & (same_type['is_workday'] == 0)]

    if len(candidates) == 0:
        return candidates

    # 找温度接近的记录
    temp_diff = np.abs(candidates['temp_avg'] - target_temp)
    candidates_with_diff = candidates.copy()
    candidates_with_diff['temp_diff'] = temp_diff

    # 计算时间接近度（天数差距）
    latest_date = df['date'].max()
    candidates_with_diff['days_ago'] = (latest_date - candidates_with_diff['date']).dt.days
    
    # 归一化：温度差异和天数差异都转为0-1分数
    # 温度容忍度：舒适区6度，其他3度
    temp_zone = get_temperature_zone(target_temp)
    temp_tolerance = 6 if temp_zone == 'comfort' else 3
    
    # 温度分数：(1 - 差异/容忍度)，最低0.2
    candidates_with_diff['temp_score'] = np.maximum(0.2, 1 - temp_diff / temp_tolerance)
    
    # 时间分数：最近7天=1.0，每增加7天降0.1，最低0.4
    candidates_with_diff['time_score'] = np.maximum(0.4, 1 - candidates_with_diff['days_ago'] / 70)
    
    # 综合分数：温度50% + 时间50%（平衡权重）
    candidates_with_diff['combined_score'] = 0.5 * candidates_with_diff['temp_score'] + 0.5 * candidates_with_diff['time_score']
    
    # 按综合分数排序
    candidates_with_diff = candidates_with_diff.sort_values('combined_score', ascending=False)
    
    return candidates_with_diff.head(15)

# 获取目标温度的工作日-周末差值（5月初高温）
target_temp = df_predict['temp_avg'].iloc[0] if len(df_predict) > 0 else 20
workday_weekend_gap = get_workday_weekend_gap_by_temp(df_history, target_temp, tolerance=3)
print(f"\n[INFO] 基于温度{target_temp:.1f}°C的工作日/周末差值: {workday_weekend_gap:.2f}")

# 备用：使用最近配对的差值（仅供参考）
def get_last_workday_weekend_pair(df):
    """获取最近的工作日和对应的周末配对数据（排除节假日）"""
    workday_result = None
    weekend_result = None
    
    for idx, row in df.iloc[::-1].iterrows():
        if row['is_holiday'] == 1:
            continue
        if row['is_workday'] == 1 and workday_result is None:
            workday_result = row['result']
        elif row['is_weekend'] == 1 and row['is_workday'] == 0 and weekend_result is None:
            weekend_result = row['result']
        if workday_result and weekend_result:
            break
    
    if workday_result and weekend_result:
        return workday_result, weekend_result, workday_result - weekend_result
    return None, None, 200

last_workday, last_weekend, last_gap = get_last_workday_weekend_pair(df_history)
if last_workday and last_weekend:
    print(f"[INFO] 历史最近配对: 工作日={last_workday:.2f}, 周末={last_weekend:.2f}, 差值={last_gap:.2f}")

for i, row in df_predict.iterrows():
    print(f"\n[INFO] 处理预测日期: {row['date'].date()}")
    
    # === 核心逻辑：查找上一个同类型日期（排除节假日） ===
    is_weekend = row['is_weekend']
    is_holiday = row['is_holiday']
    is_workday = row['is_workday']
    dayofweek = row['dayofweek']
    
    # 如果是节假日，使用前一天的预测作为基准
    if is_holiday == 1:
        base_result = predicted_results[-1] if len(predicted_results) > 0 else df_history['result'].iloc[-1]
        print(f"  [DEBUG] 节假日，跳过基准计算，使用: {base_result:.2f}")
        recent_mean = base_result
        recent_std = 0
    else:
        # 找历史上最近的上一个同类型日期（根据is_workday判断，不是is_weekend）
        # 调休日虽然is_weekend=1，但is_workday=1，应该用工作日数据
        # 找温度接近的同类型历史日期（综合考虑温度和时间）
        recent_same = find_similar_type_by_temp(df_history, row['temp_avg'], is_workday)
        
        if len(recent_same) > 0:
            recent_results = recent_same['result'].tolist()
            recent_dates = recent_same['date'].dt.strftime('%Y-%m-%d').tolist()
            recent_temps = recent_same['temp_avg'].tolist()
            recent_time_score = recent_same['time_score'].tolist() if 'time_score' in recent_same else [1.0] * len(recent_same)
            base_result = recent_results[0]  # 综合分数最高的
            recent_mean = np.mean(recent_results)
            recent_std = np.std(recent_results, ddof=1) if len(recent_results) > 1 else 0
            print(f"  [DEBUG] 最近同类型日期(综合分数):")
            for d, t, r, ts in zip(recent_dates[:3], recent_temps[:3], recent_results[:3], recent_time_score[:3]):
                print(f"    {d} (温度:{t:.1f}°C, 时间分:{ts:.2f}): {r:.2f}")
        else:
            base_result = df_history['result'].iloc[-1]
            recent_mean = base_result
            recent_std = 0
            print(f"  [DEBUG] 无历史同类型数据，使用最后结果: {base_result:.2f}")
    
    # 使用预测结果进行滚动
    if len(predicted_results) > 0:
        prev_pred = predicted_results[-1]
        # 如果前后天类型不同（工作日→周末 或 周末→工作日），考虑工作日-周末差值
        if len(predicted_results) >= 2:
            # 使用实际的工作日/周末差值进行调整
            if is_weekend == 1:  # 今天是周末
                adjusted_base = prev_pred - workday_weekend_gap
                print(f"  [DEBUG] 前一天工作日预测: {prev_pred:.2f}, 考虑工作日-周末差值: -{workday_weekend_gap:.2f} -> {adjusted_base:.2f}")
                base_result = adjusted_base
        print(f"  [DEBUG] 前一天预测值: {base_result:.2f}")
    
    # === 其次：从历史数据中查找最相似的日期（排除节假日） ===
    similarities = []
    for _, history_row in df_history.iterrows():
        if history_row['is_holiday'] == 1:  # 跳过节假日
            continue
        sim = calculate_similarity(row, history_row)
        similarities.append((sim, history_row))

    # 按相似度排序，取前15个
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_15_similar = similarities[:15]

    # 提取相似日期的result值（已排除节假日）
    similar_results = [h_row['result'] for _, h_row in top_15_similar]
    similar_dates = [h_row['date'].date() for _, h_row in top_15_similar]

    print(f"  [DEBUG] 相似日期 (Top 5, 排除节假日):")
    for idx, (sim_score, sim_date, sim_result) in enumerate(zip([s[0] for s in top_15_similar[:5]], similar_dates[:5], similar_results[:5])):
        print(f"    {sim_date} (相似度: {sim_score:.2f}): {sim_result:.2f}")

    # === 综合计算滞后特征 ===
    weights = [s[0] for s in top_15_similar]
    weights_sum = sum(weights)
    weighted_similar = sum(r * w for r, w in zip(similar_results, weights)) / weights_sum if weights_sum > 0 else recent_mean
    
    # 计算相似结果的统计信息
    similar_mean = np.mean(similar_results) if len(similar_results) > 0 else recent_mean
    similar_std = np.std(similar_results, ddof=1) if len(similar_results) > 1 else 0
    
    # 检测异常值（峰值）：超过均值+1.5倍标准差的认为是异常峰值
    # 排除峰值后重新计算，避免峰值影响
    if similar_std > 100:
        lower_bound = similar_mean - 1.5 * similar_std
        upper_bound = similar_mean + 1.5 * similar_std
        filtered_results = [r for r in similar_results if lower_bound <= r <= upper_bound]
        
        # 如果过滤后还有足够数据，使用过滤后的数据
        if len(filtered_results) >= 3:
            filtered_mean = np.mean(filtered_results)
            filtered_std = np.std(filtered_results, ddof=1) if len(filtered_results) > 1 else 0
            print(f"  [DEBUG] 检测到峰值，过滤后: {len(filtered_results)}个, 均值={filtered_mean:.2f}")
            # 使用过滤后的均值作为主要参考，降低峰值影响
            result_lag_1d = 0.5 * base_result + 0.3 * filtered_mean + 0.2 * similar_results[0]
            result_lag_2d = 0.4 * recent_mean + 0.4 * filtered_mean + 0.2 * similar_results[1]
            result_lag_3d = 0.4 * recent_mean + 0.4 * filtered_mean + 0.2 * similar_results[2]
            result_lag_7d = 0.4 * recent_mean + 0.4 * filtered_mean + 0.2 * weighted_similar
            result_roll_mean_7d = 0.5 * recent_mean + 0.5 * filtered_mean
            result_roll_std_7d = max(recent_std, filtered_std)
        else:
            # 数据不足，使用普通权重
            result_lag_1d = 0.8 * base_result + 0.2 * similar_results[0] if len(similar_results) >= 1 else base_result
            result_lag_2d = 0.6 * recent_mean + 0.4 * similar_results[1] if len(similar_results) >= 2 else result_lag_1d
            result_lag_3d = 0.5 * recent_mean + 0.5 * similar_results[2] if len(similar_results) >= 3 else result_lag_1d
            result_lag_7d = 0.5 * recent_mean + 0.5 * weighted_similar
            result_roll_mean_7d = 0.6 * recent_mean + 0.4 * weighted_similar
            result_roll_std_7d = max(recent_std, similar_std)
    else:
        # 标准差正常，使用普通权重
        result_lag_1d = 0.8 * base_result + 0.2 * similar_results[0] if len(similar_results) >= 1 else base_result
        result_lag_2d = 0.6 * recent_mean + 0.4 * similar_results[1] if len(similar_results) >= 2 else result_lag_1d
        result_lag_3d = 0.5 * recent_mean + 0.5 * similar_results[2] if len(similar_results) >= 3 else result_lag_1d
        result_lag_7d = 0.5 * recent_mean + 0.5 * weighted_similar
        result_roll_mean_7d = 0.6 * recent_mean + 0.4 * weighted_similar
        result_roll_std_7d = max(recent_std, similar_std)

    # 构建特征向量
    features = {
        'temp_min': row['temp_min'],
        'temp_avg': row['temp_avg'],
        'temp_max': row['temp_max'],
        'rainfall': row['rainfall'],
        'dayofweek': row['dayofweek'],
        'is_weekend': row['is_weekend'],
        'month': row['month'],
        'dayofyear': row['dayofyear'],
        'is_holiday': row['is_holiday'],
        'is_workday': row['is_workday'],
        'quarter': row['quarter'],
        'holiday_type': row['holiday_type'],
        'temp_diff': row['temp_diff'],
        'temp_deviation': row['temp_deviation'],
        'temp_zone': row['temp_zone'],
        'is_low_temp': row['is_low_temp'],
        'is_comfort_zone': row['is_comfort_zone'],
        'is_high_temp': row['is_high_temp'],
        'heating_load': row['heating_load'],
        'cooling_load': row['cooling_load'],
        'holiday_temp': row['holiday_temp'],
        'weekend_temp': row['weekend_temp'],
        'holiday_type_temp': row['holiday_type_temp'],
        'high_temp_holiday': row['high_temp_holiday'],
        'low_temp_holiday': row['low_temp_holiday'],
        'is_long_holiday': row['is_long_holiday'],
        'is_spring_festival': row['is_spring_festival'],
        'long_holiday_temp': row['long_holiday_temp'],
        'long_holiday_high_temp': row['long_holiday_high_temp'],
        'long_holiday_low_temp': row['long_holiday_low_temp'],
        'workday_temp': row['workday_temp'],
        'workday_high_temp': row['workday_high_temp'],
        'workday_low_temp': row['workday_low_temp'],
        'rain_impact': row['rain_impact'],
        'weather_load_factor': row['weather_load_factor'],
        'is_before_weekend': row['is_before_weekend'],
        'is_after_weekend': row['is_after_weekend'],
        # 滞后特征
        'result_lag_1d': result_lag_1d,
        'result_lag_2d': result_lag_2d,
        'result_lag_3d': result_lag_3d,
        'result_lag_7d': result_lag_7d,
        'result_roll_mean_7d': result_roll_mean_7d,
        'result_roll_std_7d': result_roll_std_7d,
    }

    # 转换为DataFrame进行预测
    X_pred = pd.DataFrame([features])[feature_cols]
    # 禁用特征数量检查以支持新增特征
    pred = model.predict(X_pred, predict_disable_shape_check=True)[0]
    predictions.append(pred)
    predicted_results.append(pred)  # 存储用于后续滚动预测

    print(f"  [OK] 预测值: {pred:.2f}")

# ==================== 9. 保存预测结果 ====================
results_df = pd.DataFrame({
    '日期': df_predict['date'].dt.strftime('%Y-%m-%d'),
    '最低温度': df_predict['temp_min'].tolist(),
    '平均温度': df_predict['temp_avg'].tolist(),
    '最高温度': df_predict['temp_max'].tolist(),
    '降雨量': df_predict['rainfall'].tolist(),
    '是否周末': df_predict['is_weekend'].tolist(),
    '是否节假日': df_predict['is_holiday'].tolist(),
    '预测值': [round(p, 2) for p in predictions],
})

output_path = r'prediction_results.csv'
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n[OK] 预测结果已保存为: {output_path}")
print("\n" + "="*50)
print("[RESULT] 预测结果汇总:")
print("="*50)
print(results_df.to_string(index=False))
print("\n[OK] 完成！")
