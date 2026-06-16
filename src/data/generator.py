import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_baseline_load(seed: int = 42) -> pd.DataFrame:
    """生成15分钟粒度的全天基线负荷预测曲线（夏季典型日形状）。

    夏季负荷特征：
    - 凌晨（0-6点）：低负荷，基础用电
    - 早高峰（7-9点）：上班开工，负荷上升
    - 午间（10-13点）：持续高位
    - 下午高峰（14-19点）：空调满负荷，全天最高
    - 晚间（20-23点）：逐渐回落

    Returns:
        DataFrame，包含 time 和 load_kw 两列，共96行
    """
    rng = np.random.default_rng(seed)

    n_points = 96
    hours = np.linspace(0, 24, n_points, endpoint=False)

    base_load = 800.0

    morning_ramp = 150.0 * (1 / (1 + np.exp(-(hours - 7.5) * 1.5)))

    afternoon_peak = 250.0 * np.exp(-((hours - 15.5) ** 2) / (2 * 3.5 ** 2))

    evening_decay = 120.0 * np.exp(-((hours - 19.0) ** 2) / (2 * 2.0 ** 2))

    noise = rng.normal(0, 10.0, n_points)

    load = base_load + morning_ramp + afternoon_peak + evening_decay + noise
    load = np.maximum(load, 500.0)

    start_time = datetime(2024, 7, 15, 0, 0)
    times = [start_time + timedelta(minutes=15 * i) for i in range(n_points)]

    df = pd.DataFrame({
        'time': times,
        'load_kw': np.round(load, 2)
    })

    return df


def generate_resource_register(seed: int = 42) -> pd.DataFrame:
    """生成柔性资源台账。

    资源类型包括：空调、冷库、充电桩、可降功率产线。
    每个资源包含：
    - resource_id: 资源唯一标识
    - name: 资源名称
    - type: 资源类型
    - max_power_kw: 可削减功率上限（kW）
    - max_duration_min: 单次最长削减时长（分钟）
    - max_calls_per_day: 一天最多调用次数
    - min_rest_min: 两次调用之间最短休息时间（分钟）
    - cost_per_kw: 单位削减成本（元/kW）
    - rebound_factor: 反弹系数（0-1，表示削减后反弹的比例）
    - rebound_duration_min: 反弹持续时间（分钟）

    Returns:
        DataFrame，柔性资源台账
    """
    rng = np.random.default_rng(seed)

    resources = []

    ac_names = ['AC-1F-东', 'AC-1F-西', 'AC-2F-办公区', 'AC-3F-会议区', 'AC-机房']
    for i, name in enumerate(ac_names):
        resources.append({
            'resource_id': f'AC{i+1:02d}',
            'name': name,
            'type': '空调',
            'max_power_kw': round(rng.uniform(30, 80), 1),
            'max_duration_min': int(rng.choice([30, 45, 60, 90])),
            'max_calls_per_day': int(rng.choice([2, 3, 4])),
            'min_rest_min': int(rng.choice([30, 45, 60])),
            'cost_per_kw': round(rng.uniform(0.8, 2.5), 2),
            'rebound_factor': round(rng.uniform(0.3, 0.7), 2),
            'rebound_duration_min': int(rng.choice([30, 45, 60, 90]))
        })

    resources.append({
        'resource_id': 'CW01',
        'name': '冷库-1号',
        'type': '冷库',
        'max_power_kw': 120.0,
        'max_duration_min': 120,
        'max_calls_per_day': 2,
        'min_rest_min': 90,
        'cost_per_kw': 1.5,
        'rebound_factor': 0.5,
        'rebound_duration_min': 60
    })
    resources.append({
        'resource_id': 'CW02',
        'name': '冷库-2号',
        'type': '冷库',
        'max_power_kw': 95.0,
        'max_duration_min': 90,
        'max_calls_per_day': 3,
        'min_rest_min': 60,
        'cost_per_kw': 1.8,
        'rebound_factor': 0.45,
        'rebound_duration_min': 45
    })

    resources.append({
        'resource_id': 'EV01',
        'name': '充电桩-A区',
        'type': '充电桩',
        'max_power_kw': 150.0,
        'max_duration_min': 180,
        'max_calls_per_day': 2,
        'min_rest_min': 120,
        'cost_per_kw': 0.5,
        'rebound_factor': 0.8,
        'rebound_duration_min': 120
    })

    pl_names = ['产线-组装', '产线-包装', '产线-测试']
    for i, name in enumerate(pl_names):
        resources.append({
            'resource_id': f'PL{i+1:02d}',
            'name': name,
            'type': '产线',
            'max_power_kw': round(rng.uniform(80, 200), 1),
            'max_duration_min': int(rng.choice([60, 90, 120])),
            'max_calls_per_day': int(rng.choice([1, 2, 3])),
            'min_rest_min': int(rng.choice([60, 90, 120])),
            'cost_per_kw': round(rng.uniform(3.0, 8.0), 2),
            'rebound_factor': round(rng.uniform(0.1, 0.3), 2),
            'rebound_duration_min': int(rng.choice([15, 30, 45]))
        })

    df = pd.DataFrame(resources)
    return df


def load_data(baseline_path: str = None, register_path: str = None) -> tuple:
    """从CSV加载数据，如果路径不存在则生成模拟数据。

    Args:
        baseline_path: 基线负荷CSV路径
        register_path: 资源台账CSV路径

    Returns:
        (baseline_df, register_df)
    """
    import os

    if baseline_path and os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        baseline_df['time'] = pd.to_datetime(baseline_df['time'])
    else:
        baseline_df = generate_baseline_load()

    if register_path and os.path.exists(register_path):
        register_df = pd.read_csv(register_path)
    else:
        register_df = generate_resource_register()

    return baseline_df, register_df


def save_data(baseline_df: pd.DataFrame, register_df: pd.DataFrame,
              baseline_path: str, register_path: str):
    """保存数据到CSV。"""
    baseline_df.to_csv(baseline_path, index=False)
    register_df.to_csv(register_path, index=False)


def validate_baseline_data(df: pd.DataFrame) -> list:
    """校验基线负荷数据，返回错误信息列表。"""
    errors = []

    if df is None or len(df) == 0:
        errors.append('基线负荷数据为空')
        return errors

    if 'time' not in df.columns:
        errors.append('基线数据缺少 time 列')
    if 'load_kw' not in df.columns:
        errors.append('基线数据缺少 load_kw 列')
        return errors

    if (df['load_kw'] < 0).any():
        neg_count = (df['load_kw'] < 0).sum()
        errors.append(f'基线负荷中有 {neg_count} 个负值，已跳过这些点')

    if len(df) != 96:
        errors.append(f'基线数据点数为 {len(df)}，预期96个（15分钟粒度全天）')

    return errors


def validate_register_data(df: pd.DataFrame) -> list:
    """校验资源台账数据，返回错误信息列表。"""
    errors = []

    if df is None or len(df) == 0:
        errors.append('资源台账数据为空')
        return errors

    required_cols = ['resource_id', 'max_power_kw', 'max_duration_min',
                     'max_calls_per_day', 'min_rest_min', 'cost_per_kw',
                     'rebound_factor', 'rebound_duration_min']
    for col in required_cols:
        if col not in df.columns:
            errors.append(f'资源台账缺少 {col} 列')

    if 'max_power_kw' in df.columns:
        neg_power = (df['max_power_kw'] <= 0).sum()
        if neg_power > 0:
            errors.append(f'有 {neg_power} 个资源的功率上限非正数，将被过滤')

    if 'min_rest_min' in df.columns:
        neg_rest = (df['min_rest_min'] < 0).sum()
        if neg_rest > 0:
            errors.append(f'有 {neg_rest} 个资源的休息间隔为负，将被修正为0')

    if 'max_duration_min' in df.columns:
        neg_dur = (df['max_duration_min'] <= 0).sum()
        if neg_dur > 0:
            errors.append(f'有 {neg_dur} 个资源的最长持续时间非正，将被过滤')

    if 'max_calls_per_day' in df.columns:
        neg_calls = (df['max_calls_per_day'] <= 0).sum()
        if neg_calls > 0:
            errors.append(f'有 {neg_calls} 个资源的日调用次数非正，将被过滤')

    if 'rebound_factor' in df.columns:
        bad_factor = ((df['rebound_factor'] < 0) | (df['rebound_factor'] > 1)).sum()
        if bad_factor > 0:
            errors.append(f'有 {bad_factor} 个资源的反弹系数不在[0,1]范围内，将被裁剪')

    return errors


def clean_register_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗资源台账数据，处理异常值。"""
    df = df.copy()

    if 'min_rest_min' in df.columns:
        df['min_rest_min'] = df['min_rest_min'].clip(lower=0)

    if 'rebound_factor' in df.columns:
        df['rebound_factor'] = df['rebound_factor'].clip(lower=0, upper=1)

    if 'max_power_kw' in df.columns:
        df = df[df['max_power_kw'] > 0].copy()

    if 'max_duration_min' in df.columns:
        df = df[df['max_duration_min'] > 0].copy()

    if 'max_calls_per_day' in df.columns:
        df = df[df['max_calls_per_day'] > 0].copy()

    return df.reset_index(drop=True)
