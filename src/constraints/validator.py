import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.rebound.model import (
    find_call_start_slots, find_call_end_slots, minutes_to_slots
)


def validate_schedule(schedule_matrix: np.ndarray,
                      register_df: pd.DataFrame,
                      slot_minutes: int = 15) -> Tuple[bool, List[str]]:
    """校验调度方案是否满足所有约束。

    校验项：
    1. 每个资源任意时刻削减量不超过功率上限
    2. 单次连续调用时长不超过上限
    3. 全天调用次数不超过上限
    4. 两次调用之间的休息间隔满足要求

    Args:
        schedule_matrix: 调度矩阵 (n_resources, n_slots)
        register_df: 资源台账
        slot_minutes: 每个时段的分钟数

    Returns:
        (是否全部通过, 错误信息列表)
    """
    errors = []
    n_resources, n_slots = schedule_matrix.shape

    for r_idx in range(n_resources):
        rid = register_df.iloc[r_idx]['resource_id']
        resource_errors = _validate_single_resource(
            r_idx, rid, schedule_matrix, register_df, slot_minutes
        )
        errors.extend(resource_errors)

    return len(errors) == 0, errors


def _validate_single_resource(r_idx: int, rid: str,
                              schedule_matrix: np.ndarray,
                              register_df: pd.DataFrame,
                              slot_minutes: int) -> List[str]:
    """校验单个资源的所有约束。"""
    errors = []
    row = register_df.iloc[r_idx]
    power_schedule = schedule_matrix[r_idx, :]

    max_power = row['max_power_kw']
    max_duration_min = row['max_duration_min']
    max_calls = int(row['max_calls_per_day'])
    min_rest_min = row['min_rest_min']

    if (power_schedule < 0).any():
        errors.append(f"资源 {rid}: 存在负的削减功率")

    over_power = np.where(power_schedule > max_power + 1e-6)[0]
    if len(over_power) > 0:
        errors.append(
            f"资源 {rid}: 有 {len(over_power)} 个时段削减功率超过上限 "
            f"({max_power} kW)，例如时段 {over_power[0]}"
        )

    active_slots = np.where(power_schedule > 1e-6)[0]
    if len(active_slots) == 0:
        return errors

    call_starts = find_call_start_slots(active_slots)
    call_ends = find_call_end_slots(active_slots)

    n_calls = len(call_starts)
    if n_calls > max_calls:
        errors.append(
            f"资源 {rid}: 调用次数 {n_calls} 超过上限 {max_calls} 次"
        )

    max_duration_slots = minutes_to_slots(max_duration_min, slot_minutes)
    for i in range(n_calls):
        duration_slots = call_ends[i] - call_starts[i] + 1
        if duration_slots > max_duration_slots:
            errors.append(
                f"资源 {rid}: 第 {i+1} 次调用持续 {duration_slots} 个时段 "
                f"({duration_slots * slot_minutes} 分钟)，"
                f"超过上限 {max_duration_min} 分钟"
            )

    min_rest_slots = minutes_to_slots(min_rest_min, slot_minutes)
    for i in range(1, n_calls):
        rest_slots = call_starts[i] - call_ends[i-1] - 1
        if rest_slots < min_rest_slots:
            errors.append(
                f"资源 {rid}: 第 {i} 次和第 {i+1} 次调用之间休息 "
                f"{rest_slots} 个时段 ({rest_slots * slot_minutes} 分钟)，"
                f"不足 {min_rest_min} 分钟"
            )

    return errors


def check_peak_constraint(load_curve: np.ndarray,
                          target_peak: float) -> Tuple[bool, List[int], float]:
    """检查负荷曲线是否满足峰值约束。

    Args:
        load_curve: 负荷曲线
        target_peak: 目标峰值

    Returns:
        (是否满足, 超限时段列表, 总超限量)
    """
    over_slots = np.where(load_curve > target_peak + 1e-6)[0]
    total_deficit = float(np.sum(np.maximum(load_curve - target_peak, 0)))
    return len(over_slots) == 0, over_slots.tolist(), total_deficit


def count_resource_calls(schedule_matrix: np.ndarray) -> List[int]:
    """统计每个资源的调用次数。

    Args:
        schedule_matrix: 调度矩阵

    Returns:
        每个资源的调用次数列表
    """
    n_resources = schedule_matrix.shape[0]
    counts = []
    for r_idx in range(n_resources):
        active_slots = np.where(schedule_matrix[r_idx, :] > 1e-6)[0]
        if len(active_slots) == 0:
            counts.append(0)
            continue
        count = 1
        prev = active_slots[0]
        for s in active_slots[1:]:
            if s != prev + 1:
                count += 1
            prev = s
        counts.append(count)
    return counts


def get_call_intervals(schedule_matrix: np.ndarray,
                       r_idx: int) -> List[Tuple[int, int]]:
    """获取单个资源的所有调用区间。

    Args:
        schedule_matrix: 调度矩阵
        r_idx: 资源索引

    Returns:
        [(start_slot, end_slot), ...] 列表
    """
    active_slots = np.where(schedule_matrix[r_idx, :] > 1e-6)[0]
    if len(active_slots) == 0:
        return []

    intervals = []
    start = active_slots[0]
    prev = active_slots[0]
    for s in active_slots[1:]:
        if s != prev + 1:
            intervals.append((int(start), int(prev)))
            start = s
        prev = s
    intervals.append((int(start), int(prev)))
    return intervals
