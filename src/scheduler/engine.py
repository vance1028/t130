import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class ResourceState:
    """单个资源的调度状态跟踪。"""
    resource_id: str
    max_power_kw: float
    max_duration_slots: int
    max_calls_per_day: int
    min_rest_slots: int
    cost_per_kw: float
    rebound_factor: float
    rebound_duration_slots: int

    calls_count: int = 0
    current_streak: int = 0
    last_end_slot: int = -9999
    is_active: bool = False

    def can_start(self, slot: int) -> bool:
        """判断在slot时段是否可以开始新的调用。"""
        if self.calls_count >= self.max_calls_per_day:
            return False
        if self.is_active:
            return True
        rest_slots = slot - self.last_end_slot - 1
        return rest_slots >= self.min_rest_slots

    def can_continue(self) -> bool:
        """判断是否可以继续当前调用。"""
        if not self.is_active:
            return False
        return self.current_streak < self.max_duration_slots

    def start_call(self, slot: int):
        """开始一次新的调用。"""
        self.is_active = True
        self.current_streak = 1
        self.calls_count += 1

    def continue_call(self):
        """继续当前调用（延长一个时段）。"""
        self.current_streak += 1

    def end_call(self, slot: int):
        """结束当前调用。"""
        self.is_active = False
        self.last_end_slot = slot
        self.current_streak = 0


@dataclass
class ScheduleResult:
    """调度结果。"""
    success: bool
    baseline_load: np.ndarray
    target_peak: float
    scheduled_load: np.ndarray
    schedule_matrix: np.ndarray
    rebound_load: np.ndarray
    total_cost: float
    resource_calls: Dict[str, int]
    peak_after_schedule: float
    peak_before_schedule: float
    total_curtailment: float
    total_rebound: float
    deficit_slots: List[int]
    deficit_amount: float
    resource_states: Dict[str, ResourceState] = field(default_factory=dict)


def _minutes_to_slots(minutes: int, slot_minutes: int = 15) -> int:
    """分钟数转成时段数（向上取整）。"""
    return int(np.ceil(minutes / slot_minutes))


def _create_resource_states(register_df: pd.DataFrame) -> Dict[str, ResourceState]:
    """从资源台账创建资源状态对象。"""
    states = {}
    for _, row in register_df.iterrows():
        rid = row['resource_id']
        states[rid] = ResourceState(
            resource_id=rid,
            max_power_kw=row['max_power_kw'],
            max_duration_slots=_minutes_to_slots(row['max_duration_min']),
            max_calls_per_day=int(row['max_calls_per_day']),
            min_rest_slots=_minutes_to_slots(row['min_rest_min']),
            cost_per_kw=row['cost_per_kw'],
            rebound_factor=row['rebound_factor'],
            rebound_duration_slots=_minutes_to_slots(row['rebound_duration_min'])
        )
    return states


def _get_sorted_resources(states: Dict[str, ResourceState],
                          descending: bool = False) -> List[str]:
    """按单位成本排序资源ID列表（默认升序：便宜的在前）。"""
    return sorted(states.keys(),
                  key=lambda r: states[r].cost_per_kw,
                  reverse=descending)


def _find_call_end_slots(active_slots: np.ndarray) -> List[int]:
    """从活动时段数组中找出每次调用的结束时段。"""
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


def _count_calls(active_slots: np.ndarray) -> int:
    """统计调用次数（连续时段算一次调用）。"""
    if len(active_slots) == 0:
        return 0
    count = 1
    prev = active_slots[0]
    for s in active_slots[1:]:
        if s != prev + 1:
            count += 1
        prev = s
    return count


def _avg_power_of_call(power_schedule: np.ndarray, end_slot: int,
                       max_duration: int) -> float:
    """计算一次调用的平均功率。"""
    start_slot = max(0, end_slot - max_duration + 1)
    while start_slot < end_slot and power_schedule[start_slot] == 0:
        start_slot += 1
    call_powers = power_schedule[start_slot:end_slot + 1]
    if len(call_powers) == 0:
        return 0.0
    return float(np.mean(call_powers))


def _compute_full_rebound(schedule_matrix: np.ndarray,
                          states: Dict[str, ResourceState],
                          resource_ids: List[str],
                          n_slots: int) -> np.ndarray:
    """计算完整的反弹负荷曲线。"""
    rebound_load = np.zeros(n_slots)

    for r_idx, rid in enumerate(resource_ids):
        state = states[rid]
        power_schedule = schedule_matrix[r_idx, :]
        active_slots = np.where(power_schedule > 0)[0]

        if len(active_slots) == 0:
            continue

        call_end_slots = _find_call_end_slots(active_slots)

        for end_slot in call_end_slots:
            avg_power = _avg_power_of_call(power_schedule, end_slot,
                                           state.max_duration_slots)
            rebound_slots = state.rebound_duration_slots
            rebound_factor = state.rebound_factor

            for i in range(rebound_slots):
                rebound_slot = end_slot + 1 + i
                if rebound_slot >= n_slots:
                    break
                ratio = 1.0 - (i / rebound_slots)
                rebound_load[rebound_slot] += avg_power * rebound_factor * ratio

    return rebound_load


def _calculate_total_cost(schedule_matrix: np.ndarray,
                          states: Dict[str, ResourceState],
                          resource_ids: List[str]) -> float:
    """计算总削减成本。"""
    total = 0.0
    for r_idx, rid in enumerate(resource_ids):
        total += np.sum(schedule_matrix[r_idx, :]) * states[rid].cost_per_kw
    return float(total)


def _schedule_by_time_order(baseline_load: np.ndarray,
                            target_peak: float,
                            register_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, ResourceState]]:
    """按时段从左到右扫描的贪心调度。

    策略：
    1. 从左到右扫描每个时段
    2. 对每个超峰时段，先让已运行的资源继续运行
    3. 还不够的话，按成本从低到高启动新资源
    4. 如果某个时段不超峰了，所有正在运行的资源都停止
       （保证下次遇到峰时还能重新启动，且节省成本）

    为了提高资源利用率，我们做了一个优化：
    如果一个资源刚启动不久，虽然当前时段不超峰了，
    但后面很快又会超峰，我们可以让它继续运行以避免频繁启停。
    不过为了简单，先实现基础版本。
    """
    n_slots = len(baseline_load)
    n_resources = len(register_df)
    resource_ids = register_df['resource_id'].tolist()

    states = _create_resource_states(register_df)
    schedule_matrix = np.zeros((n_resources, n_slots))

    deficit = np.maximum(baseline_load - target_peak, 0.0)

    for slot in range(n_slots):
        if deficit[slot] <= 1e-6:
            for rid in resource_ids:
                state = states[rid]
                if state.is_active:
                    state.end_call(slot - 1)
            continue

        remaining_deficit = deficit[slot]

        sorted_resources = _get_sorted_resources(states)

        for rid in sorted_resources:
            if remaining_deficit <= 1e-6:
                break

            state = states[rid]
            r_idx = resource_ids.index(rid)

            if state.is_active:
                if state.can_continue():
                    curtail = min(state.max_power_kw, remaining_deficit)
                    schedule_matrix[r_idx, slot] = curtail
                    remaining_deficit -= curtail
                    state.continue_call()
                continue
            else:
                if state.can_start(slot):
                    state.start_call(slot)
                    curtail = min(state.max_power_kw, remaining_deficit)
                    schedule_matrix[r_idx, slot] = curtail
                    remaining_deficit -= curtail
                    continue

        for rid in resource_ids:
            state = states[rid]
            r_idx = resource_ids.index(rid)
            if state.is_active and schedule_matrix[r_idx, slot] <= 1e-6:
                state.end_call(slot - 1)

    for rid in resource_ids:
        state = states[rid]
        if state.is_active:
            state.end_call(n_slots - 1)

    final_states = _create_resource_states(register_df)
    for r_idx, rid in enumerate(resource_ids):
        active_slots = np.where(schedule_matrix[r_idx, :] > 1e-6)[0]
        if len(active_slots) > 0:
            final_states[rid].calls_count = _count_calls(active_slots)
            final_states[rid].last_end_slot = int(active_slots[-1])

    return schedule_matrix, final_states


def greedy_schedule_without_rebound(baseline_load: np.ndarray,
                                    target_peak: float,
                                    register_df: pd.DataFrame) -> ScheduleResult:
    """不考虑反弹的贪心削峰调度。

    多轮迭代的时间顺序贪心算法，确保充分利用资源。
    """
    n_slots = len(baseline_load)
    n_resources = len(register_df)
    resource_ids = register_df['resource_id'].tolist()

    schedule_matrix, final_states = _schedule_by_time_order(
        baseline_load, target_peak, register_df
    )

    scheduled_load = baseline_load - schedule_matrix.sum(axis=0)
    rebound_load = np.zeros(n_slots)

    total_cost = _calculate_total_cost(schedule_matrix, final_states, resource_ids)

    resource_calls = {}
    for r_idx, rid in enumerate(resource_ids):
        active_slots = np.where(schedule_matrix[r_idx, :] > 1e-6)[0]
        resource_calls[rid] = _count_calls(active_slots)

    peak_after = float(np.max(scheduled_load))
    peak_before = float(np.max(baseline_load))
    total_curtailment_val = float(np.sum(schedule_matrix))
    total_rebound_val = 0.0

    deficit_slots = np.where(scheduled_load > target_peak + 1e-6)[0].tolist()
    deficit_amount = float(np.sum(np.maximum(scheduled_load - target_peak, 0)))

    success = len(deficit_slots) == 0

    return ScheduleResult(
        success=success,
        baseline_load=baseline_load.copy(),
        target_peak=target_peak,
        scheduled_load=scheduled_load,
        schedule_matrix=schedule_matrix,
        rebound_load=rebound_load,
        total_cost=total_cost,
        resource_calls=resource_calls,
        peak_after_schedule=peak_after,
        peak_before_schedule=peak_before,
        total_curtailment=total_curtailment_val,
        total_rebound=total_rebound_val,
        deficit_slots=deficit_slots,
        deficit_amount=deficit_amount,
        resource_states=final_states
    )


def greedy_schedule_with_rebound(baseline_load: np.ndarray,
                                 target_peak: float,
                                 register_df: pd.DataFrame) -> ScheduleResult:
    """考虑反弹的贪心削峰调度。

    迭代优化算法：
    1. 先做一轮不考虑反弹的调度
    2. 计算反弹对负荷的影响
    3. 将反弹叠加到有效负荷上，重新调度
    4. 重复迭代，直到收敛或达到最大迭代次数
    """
    n_slots = len(baseline_load)
    n_resources = len(register_df)
    resource_ids = register_df['resource_id'].tolist()

    effective_load = baseline_load.copy()
    best_schedule = None
    best_peak = float('inf')

    max_iterations = 20

    for iteration in range(max_iterations):
        schedule_matrix, states = _schedule_by_time_order(
            effective_load, target_peak, register_df
        )

        rebound_load = _compute_full_rebound(schedule_matrix, states,
                                             resource_ids, n_slots)
        actual_load = baseline_load - schedule_matrix.sum(axis=0) + rebound_load
        actual_peak = float(np.max(actual_load))

        if actual_peak < best_peak - 1e-6:
            best_peak = actual_peak
            best_schedule = (schedule_matrix.copy(), states, rebound_load.copy())

        new_effective_load = baseline_load - schedule_matrix.sum(axis=0) + rebound_load

        if np.allclose(new_effective_load, effective_load, atol=1e-3):
            effective_load = new_effective_load
            break

        effective_load = new_effective_load

    if best_schedule is not None:
        schedule_matrix, states, rebound_load = best_schedule
    else:
        schedule_matrix = np.zeros((n_resources, n_slots))
        states = _create_resource_states(register_df)
        rebound_load = np.zeros(n_slots)

    scheduled_load = baseline_load - schedule_matrix.sum(axis=0) + rebound_load

    total_cost = _calculate_total_cost(schedule_matrix, states, resource_ids)

    resource_calls = {}
    for r_idx, rid in enumerate(resource_ids):
        active_slots = np.where(schedule_matrix[r_idx, :] > 1e-6)[0]
        resource_calls[rid] = _count_calls(active_slots)

    peak_after = float(np.max(scheduled_load))
    peak_before = float(np.max(baseline_load))
    total_curtailment_val = float(np.sum(schedule_matrix))
    total_rebound_val = float(np.sum(rebound_load))

    deficit_slots = np.where(scheduled_load > target_peak + 1e-6)[0].tolist()
    deficit_amount = float(np.sum(np.maximum(scheduled_load - target_peak, 0)))

    success = len(deficit_slots) == 0

    final_states = _create_resource_states(register_df)
    for r_idx, rid in enumerate(resource_ids):
        active_slots = np.where(schedule_matrix[r_idx, :] > 1e-6)[0]
        if len(active_slots) > 0:
            final_states[rid].calls_count = _count_calls(active_slots)
            final_states[rid].last_end_slot = int(active_slots[-1])

    return ScheduleResult(
        success=success,
        baseline_load=baseline_load.copy(),
        target_peak=target_peak,
        scheduled_load=scheduled_load,
        schedule_matrix=schedule_matrix,
        rebound_load=rebound_load,
        total_cost=total_cost,
        resource_calls=resource_calls,
        peak_after_schedule=peak_after,
        peak_before_schedule=peak_before,
        total_curtailment=total_curtailment_val,
        total_rebound=total_rebound_val,
        deficit_slots=deficit_slots,
        deficit_amount=deficit_amount,
        resource_states=final_states
    )
