# lightgbm_power
lightgbm模型预测未来期望数据
# 预测模型 - 特征参数说明

## 项目概述

本项目使用LightGBM机器学习模型进行预测，综合考虑天气、时间、节假日和季节性等多种因素。

## 核心特征参数分类

### 1. 基础天气特征 (5个)
- `temp_min`: 最低温度
- `temp_avg`: 平均温度  
- `temp_max`: 最高温度
- `rainfall`: 降雨量
- `temp_diff`: 温差 (最高温 - 最低温)

### 2. 时间特征 (7个)
- `dayofweek`: 星期几 (0=周一, 6=周日)
- `is_weekend`: 是否周末 (0/1)
- `month`: 月份
- `dayofyear`: 一年中的第几天
- `quarter`: 季度
- `is_before_weekend`: 是否为周五 (周末前一天)
- `is_after_weekend`: 是否为周一 (周末后一天)

### 3. 节假日特征 (8个)
- `is_holiday`: 是否法定节假日
- `is_workday`: 是否工作日 (考虑调休)
- `holiday_type`: 节假日类型
  - 0: 普通工作日
  - 1: 普通周末
  - 2: 短期假期(3天): 元旦、清明、端午、中秋
  - 3: 长期假期(5天): 五一、国庆
  - 4: 春节(7天)
- `is_workday_adjustment`: 是否调休工作日
- `holiday_temp`: 节假日温度交互项
- `weekend_temp`: 周末温度交互项
- `holiday_type_temp`: 节假日类型与温度交互项

### 4. 温度区间特征 (9个)
- `temp_zone`: 温度区间
  - 0: 低温区 (<15°C) - 采暖负荷明显
  - 1: 舒适区 (15-25°C) - 空调负荷低
  - 2: 高温区 (>25°C) - 制冷负荷明显
- `is_low_temp`: 是否低温区
- `is_comfort_zone`: 是否舒适区
- `is_high_temp`: 是否高温区
- `heating_load`: 采暖负荷指数 = max(0, 15 - temp_avg)
- `cooling_load`: 制冷负荷指数 = max(0, temp_avg - 25)
- `high_temp_holiday`: 高温×节假日交互项
- `low_temp_holiday`: 低温×节假日交互项

### 5. 铁路运载相关特征 (6个)
- `is_long_holiday`: 是否长假期 (五一、国庆、春节)
- `is_spring_festival`: 是否春节
- `long_holiday_temp`: 长假期温度交互项
- `long_holiday_high_temp`: 长假期高温交互项
- `long_holiday_low_temp`: 长假期低温交互项

### 6. 工作日负荷特征 (6个)
- `workday_temp`: 工作日温度交互项
- `workday_high_temp`: 工作日高温交互项
- `workday_low_temp`: 工作日低温交互项
- `rain_impact`: 降雨对工作日影响
- `weather_load_factor`: 天气综合影响因子
- `temp_deviation`: 温度偏离度 (|temp_avg - 20|)

### 7. 季节性上升期特征 (5个)
- `seasonal_rising`: 季节性上升强度 (0-1)
  - 反映电力负荷的系统性上升/下降趋势
  - 五一后恢复期: 0.9
  - 夏季高温期: 0.8-0.9
  - 冬季采暖期: 0.7-0.8
  - 平稳期: 0.5
- `rising_temp`: 季节性上升×温度交互项
- `rising_holiday`: 季节性上升×节假日交互项
- `rising_workday`: 季节性上升×工作日交互项
- `is_post_holiday_rise`: 是否节后恢复期

### 8. 滞后特征 (6个)
基于历史相似日期的结果构建：
- `result_lag_1d`: 前1天预测值 (加权组合)
- `result_lag_2d`: 前2天预测值
- `result_lag_3d`: 前3天预测值
- `result_lag_7d`: 前7天预测值
- `result_roll_mean_7d`: 7天滚动均值
- `result_roll_std_7d`: 7天滚动标准差

## 特征工程亮点

### 1. 智能相似度计算
- 温度区间匹配权重: 35%
- 节假日类型匹配权重: 30%
- 季节性时期匹配权重: 25%
- 时间接近度权重: 15%
- 降雨量匹配权重: 10%

### 2. 工作日-周末差异建模
- 根据温度范围动态调整工作日与周末的负荷差值
- 舒适区容忍度更大(±8°C)，极端温度区容忍度小(±5°C)

### 3. 异常值处理
- 自动检测并过滤超过1.5倍标准差的峰值数据
- 保证滞后特征的稳定性

### 4. 季节性周期建模
- 识别10种不同的季节性模式
- 捕捉春节前后、五一恢复期、夏季高峰等关键时期

## 模型配置

```python
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
```

## 输入数据要求

### 历史数据文件 (tset.csv)
必需字段：
- `date`: 日期 (YYYY-MM-DD格式)
- `temp_min`, `temp_avg`, `temp_max`: 温度数据
- `rainfall`: 降雨量
- `result`: 实际电力负荷值

### 预测天气数据 (tmp.csv)
必需字段：
- `date`: 预测日期
- `temp_min`, `temp_avg`, `temp_max`: 预测温度
- `rainfall`: 预测降雨量

## 输出结果

生成 `prediction_results.csv`，包含：
- 日期
- 温度信息 (最低、平均、最高)
- 降雨量
- 是否周末/节假日标识
- **预测电力负荷值**

## 注意事项

1. 模型针对2026年中国节假日进行了专门配置
2. 调休日已纳入工作日考虑范围
3. 春节、国庆等长假期对铁路电力负荷有显著影响
4. 温度区间比绝对温度值更重要
5. 季节性趋势在预测中占有重要权重
