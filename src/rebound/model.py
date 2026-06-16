import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def minutes_to_slots(minutes: int, slot_minutes: int = 15) -> int:
    """分钟数转时段数（向上取整）。"""
    return int(np.ceil(minutes / slot_minutes))


def find_call_end_slots(active_slots: np.ndarray) -> List[int]:
    """从活动时段数组中找出每次调用的结束时段。

    连续的活动时段算一次调用。

    Args:
        active_slots: 活动时段的索引数组（已排序）

    Returns:
        每次调用的结束时段索引列表
    """
    if len(active_slots) == 0:
        return []

    ends = []
    prev = active_slots[0]
    for s in active_slots[1:]:
        if s != prev + 1:
            ends.append(int(prev))
        prev = s
    ends.append(int(prev))
    return ends


def find_call_start_slots(active_slots: np.ndarray) -> List[int]:
    """从活动时段数组中找出每次调用的开始时段。"""
    if len(active_slots) == 0:
        return []

    starts = [int(active_slots[0])]
    prev = active_slots[0]
    for s in active_slots[1:]:
        if s != prev + 1:
            starts.append(int(s))
        prev = s
    return starts


def avg_power_of_call(power_schedule: np.ndarray, end_slot: int,
                      max_duration_slots: int) -> float:
    """计算一次调用的平均削减功率。

    Args:
        power_schedule: 单个资源的功率调度数组
        end_slot: 调用的结束时段
        max_duration_slots: 最大持续时段数（用于回溯查找开始）

    Returns:
        平均削减功率（kW）
    """
    start_slot = max(0, end_slot - max_duration_slots + 1)
    while start_slot < end_slot and power_schedule[start_slot] == 0:
        start_slot += 1
    call_powers = power_schedule[start_slot:end_slot + 1]
    if len(call_powers) == 0:
        return 0.0
    return float(np.mean(call_powers))


def compute_rebound_single_resource(power_schedule: np.ndarray,
                                    rebound_factor: float,
                                    rebound_duration_min: int,
                                    slot_minutes: int = 15) -> np.ndarray:
    """计算单个资源的反弹负荷曲线。

    反弹模式：
    - 每次调用结束后开始反弹
    - 反弹功率线性衰减（从最大到0）
    - 最大反弹功率 = 平均削减功率 × 反弹系数
    - 反弹持续时间由参数指定

    Args:
        power_schedule: 资源的功率调度数组（每个时段的削减功率）
        rebound_factor: 反弹系数（0-1）
        rebound_duration_min: 反弹持续时间（分钟）
        slot_minutes: 单个时段的分钟数

    Returns:
        反弹负荷数组，长度与输入相同
    """
    n_slots = len(power_schedule)
    rebound_load = np.zeros(n_slots)

    active_slots = np.where(power_schedule > 0)[0]
    if len(active_slots) == 0:
        return rebound_load

    rebound_slots = minutes_to_slots(rebound_duration_min, slot_minutes)
    call_end_slots = find_call_end_slots(active_slots)

    for end_slot in call_end_slots:
        avg_p = avg_power_of_call(power_schedule, end_slot,
                                  max_duration_slots=rebound_slots * 2)
        peak_rebound = avg_p * rebound_factor

        for i in range(rebound_slots):
            rebound_slot = end_slot + 1 + i
            if rebound_slot >= n_slots:
                break
            ratio = 1.0 - (i / rebound_slots)
            rebound_load[rebound_slot] += peak_rebound * ratio

    return rebound_load


def compute_rebound_all_resources(schedule_matrix: np.ndarray,
                                  register_df: pd.DataFrame) -> np.ndarray:
    """计算所有资源的总反弹负荷。

    Args:
        schedule_matrix: 调度矩阵，shape = (n_resources, n_slots)
        register_df: 资源台账DataFrame

    Returns:
        总反弹负荷数组
    """
    n_resources, n_slots = schedule_matrix.shape
    total_rebound = np.zeros(n_slots)

    for r_idx in range(n_resources):
        row = register_df.iloc[r_idx]
        rebound = compute_rebound_single_resource(
            schedule_matrix[r_idx, :],
            rebound_factor=row['rebound_factor'],
            rebound_duration_min=row['rebound_duration_min']
        )
        total_rebound += rebound

    return total_rebound


def compute_net_load(baseline_load: np.ndarray,
                     schedule_matrix: np.ndarray,
                     register_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """计算考虑反弹后的净负荷曲线。

    Args:
        baseline_load: 基线负荷数组
        schedule_matrix: 调度矩阵
        register_df: 资源台账

    Returns:
        (净负荷数组, 反弹负荷数组)
    """
    total_curtailment = schedule_matrix.sum(axis=0)
    rebound_load = compute_rebound_all_resources(schedule_matrix, register_df)
    net_load = baseline_load - total_curtailment + rebound_load
    return net_load, rebound_load


def summarize_rebound_effect(baseline_load: np.ndarray,
                             schedule_matrix: np.ndarray,
                             register_df: pd.DataFrame) -> Dict:
    """总结反弹效应的影响。

    Args:
        baseline_load: 基线负荷
        schedule_matrix: 调度矩阵
        register_df: 资源台账

    Returns:
        包含反弹统计信息的字典
    """
    n_slots = len(baseline_load)
    total_curtailment = schedule_matrix.sum(axis=0)
    rebound_load = compute_rebound_all_resources(schedule_matrix, register_df)

    gross_curtailment = float(np.sum(total_curtailment))
    total_rebound = float(np.sum(rebound_load))
    net_curtailment = gross_curtailment - total_rebound
    rebound_ratio = total_rebound / gross_curtailment if gross_curtailment > 0 else 0.0

    load_without_rebound = baseline_load - total_curtailment
    load_with_rebound = baseline_load - total_curtailment + rebound_load

    peak_without_rebound = float(np.max(load_without_rebound))
    peak_with_rebound = float(np.max(load_with_rebound))
    peak_increase = peak_with_rebound - peak_without_rebound

    rebound_peak_slot = int(np.argmax(rebound_load))
    rebound_peak_value = float(np.max(rebound_load))

    return {
        'gross_curtailment_kwh': gross_curtailment * 0.25,
        'total_rebound_kwh': total_rebound * 0.25,
        'net_curtailment_kwh': net_curtailment * 0.25,
        'rebound_ratio': rebound_ratio,
        'peak_without_rebound_kw': peak_without_rebound,
        'peak_with_rebound_kw': peak_with_rebound,
        'peak_increase_due_to_rebound_kw': peak_increase,
        'rebound_peak_slot': rebound_peak_slot,
        'rebound_peak_value_kw': rebound_peak_value
    }
