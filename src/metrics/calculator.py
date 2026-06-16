import numpy as np
import pandas as pd
from typing import Dict, Any
from src.rebound.model import compute_rebound_all_resources


def compute_schedule_metrics(baseline_load: np.ndarray,
                             schedule_matrix: np.ndarray,
                             register_df: pd.DataFrame,
                             target_peak: float,
                             slot_minutes: int = 15) -> Dict[str, Any]:
    """计算调度方案的各项指标。

    Args:
        baseline_load: 基线负荷数组
        schedule_matrix: 调度矩阵 (n_resources, n_slots)
        register_df: 资源台账
        target_peak: 目标峰值
        slot_minutes: 单个时段的分钟数

    Returns:
        包含所有指标的字典
    """
    n_slots = len(baseline_load)
    n_resources = len(register_df)

    total_curtailment = schedule_matrix.sum(axis=0)
    rebound_load = compute_rebound_all_resources(schedule_matrix, register_df)
    net_load = baseline_load - total_curtailment + rebound_load

    peak_before = float(np.max(baseline_load))
    peak_after = float(np.max(net_load))

    peak_reduction = peak_before - peak_after
    peak_reduction_pct = peak_reduction / peak_before if peak_before > 0 else 0.0

    over_target = net_load > target_peak + 1e-6
    n_over_slots = int(np.sum(over_target))
    total_deficit = float(np.sum(np.maximum(net_load - target_peak, 0)))
    meets_target = n_over_slots == 0

    total_curtailment_energy = float(np.sum(total_curtailment)) * (slot_minutes / 60.0)
    total_rebound_energy = float(np.sum(rebound_load)) * (slot_minutes / 60.0)
    net_curtailment_energy = total_curtailment_energy - total_rebound_energy

    costs = np.zeros(n_resources)
    for r_idx in range(n_resources):
        cost_per_kw = register_df.iloc[r_idx]['cost_per_kw']
        resource_curtail = schedule_matrix[r_idx, :].sum()
        costs[r_idx] = resource_curtail * cost_per_kw * (slot_minutes / 60.0)

    total_cost = float(np.sum(costs))

    avg_cost_per_kw = total_cost / total_curtailment_energy if total_curtailment_energy > 0 else 0.0

    resource_metrics = []
    for r_idx in range(n_resources):
        row = register_df.iloc[r_idx]
        rid = row['resource_id']
        resource_total = float(np.sum(schedule_matrix[r_idx, :]))
        resource_energy = resource_total * (slot_minutes / 60.0)
        resource_cost = float(costs[r_idx])
        resource_calls = _count_calls(schedule_matrix[r_idx, :])
        max_power = row['max_power_kw']
        utilization = resource_total / (max_power * n_slots) if max_power > 0 else 0.0

        resource_metrics.append({
            'resource_id': rid,
            'name': row['name'],
            'type': row['type'],
            'total_curtailment_kwh': round(resource_energy, 2),
            'total_cost': round(resource_cost, 2),
            'call_count': resource_calls,
            'max_calls_allowed': int(row['max_calls_per_day']),
            'utilization_ratio': round(utilization, 4),
            'cost_per_kwh': round(resource_cost / resource_energy, 2) if resource_energy > 0 else 0
        })

    rebound_peak_idx = int(np.argmax(rebound_load))
    rebound_peak_val = float(np.max(rebound_load))

    return {
        'peak_before_kw': round(peak_before, 2),
        'peak_after_kw': round(peak_after, 2),
        'peak_reduction_kw': round(peak_reduction, 2),
        'peak_reduction_pct': round(peak_reduction_pct * 100, 2),
        'target_peak_kw': target_peak,
        'meets_target': meets_target,
        'over_target_slots': n_over_slots,
        'total_deficit_kwh': round(total_deficit * (slot_minutes / 60.0), 2),
        'total_curtailment_kwh': round(total_curtailment_energy, 2),
        'total_rebound_kwh': round(total_rebound_energy, 2),
        'net_curtailment_kwh': round(net_curtailment_energy, 2),
        'rebound_ratio': round(total_rebound_energy / total_curtailment_energy, 4) if total_curtailment_energy > 0 else 0,
        'total_cost': round(total_cost, 2),
        'avg_cost_per_kwh': round(avg_cost_per_kw, 2),
        'rebound_peak_slot': rebound_peak_idx,
        'rebound_peak_value_kw': round(rebound_peak_val, 2),
        'resource_metrics': resource_metrics
    }


def _count_calls(power_schedule: np.ndarray) -> int:
    """统计单个资源的调用次数。"""
    active_slots = np.where(power_schedule > 1e-6)[0]
    if len(active_slots) == 0:
        return 0
    count = 1
    prev = active_slots[0]
    for s in active_slots[1:]:
        if s != prev + 1:
            count += 1
        prev = s
    return count


def compare_schedules(baseline_load: np.ndarray,
                      schedule_no_rebound: np.ndarray,
                      schedule_with_rebound: np.ndarray,
                      register_df: pd.DataFrame,
                      target_peak: float) -> Dict[str, Any]:
    """对比考虑反弹和不考虑反弹两种调度方案。

    Args:
        baseline_load: 基线负荷
        schedule_no_rebound: 不考虑反弹的调度矩阵
        schedule_with_rebound: 考虑反弹的调度矩阵
        register_df: 资源台账
        target_peak: 目标峰值

    Returns:
        对比结果字典
    """
    metrics_no_rebound = compute_schedule_metrics(
        baseline_load, schedule_no_rebound, register_df, target_peak
    )
    metrics_with_rebound = compute_schedule_metrics(
        baseline_load, schedule_with_rebound, register_df, target_peak
    )

    rebound_no = compute_rebound_all_resources(schedule_no_rebound, register_df)
    actual_peak_no_rebound = float(np.max(baseline_load - schedule_no_rebound.sum(axis=0) + rebound_no))

    comparison = {
        'no_rebound_simulation': {
            'peak_after_schedule_kw': metrics_no_rebound['peak_after_kw'],
            'total_cost': metrics_no_rebound['total_cost'],
            'total_curtailment_kwh': metrics_no_rebound['total_curtailment_kwh'],
            'meets_target_in_simulation': metrics_no_rebound['meets_target']
        },
        'with_rebound_simulation': {
            'peak_after_schedule_kw': metrics_with_rebound['peak_after_kw'],
            'total_cost': metrics_with_rebound['total_cost'],
            'total_curtailment_kwh': metrics_with_rebound['total_curtailment_kwh'],
            'meets_target': metrics_with_rebound['meets_target']
        },
        'naive_schedule_actual_peak_with_rebound_kw': round(actual_peak_no_rebound, 2),
        'peak_difference_kw': round(metrics_with_rebound['peak_after_kw'] - actual_peak_no_rebound, 2),
        'cost_difference': round(metrics_with_rebound['total_cost'] - metrics_no_rebound['total_cost'], 2)
    }

    return comparison
