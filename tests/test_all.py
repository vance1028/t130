import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.scheduler.engine import (
    greedy_schedule_with_rebound,
    greedy_schedule_without_rebound
)
from src.constraints.validator import (
    validate_schedule,
    check_peak_constraint,
    count_resource_calls
)
from src.rebound.model import (
    compute_rebound_single_resource,
    compute_rebound_all_resources,
    compute_net_load,
    find_call_end_slots,
    find_call_start_slots
)


def _create_simple_register():
    """创建一个简单的资源台账用于测试。"""
    data = [
        {
            'resource_id': 'R001',
            'name': '测试资源1',
            'type': '空调',
            'max_power_kw': 100.0,
            'max_duration_min': 60,
            'max_calls_per_day': 3,
            'min_rest_min': 30,
            'cost_per_kw': 1.0,
            'rebound_factor': 0.5,
            'rebound_duration_min': 30
        },
        {
            'resource_id': 'R002',
            'name': '测试资源2',
            'type': '冷库',
            'max_power_kw': 150.0,
            'max_duration_min': 90,
            'max_calls_per_day': 2,
            'min_rest_min': 60,
            'cost_per_kw': 2.0,
            'rebound_factor': 0.3,
            'rebound_duration_min': 60
        },
    ]
    return pd.DataFrame(data)


def _create_simple_baseline(n_slots=96):
    """创建一个简单的基线负荷曲线。"""
    baseline = np.ones(n_slots) * 500.0

    peak_start = 40
    peak_end = 60
    for i in range(peak_start, peak_end):
        baseline[i] = 800.0

    return baseline


class TestReboundModel:
    """反弹负荷建模测试。"""

    def test_single_call_rebound(self):
        """测试单次调用的反弹负荷计算。"""
        n_slots = 20
        power_schedule = np.zeros(n_slots)
        power_schedule[2:6] = 100.0

        rebound = compute_rebound_single_resource(
            power_schedule,
            rebound_factor=0.5,
            rebound_duration_min=45,
            slot_minutes=15
        )

        assert len(rebound) == n_slots
        assert rebound[6] > 0
        assert rebound[5] == pytest.approx(0)
        assert rebound[8] < rebound[6]
        assert rebound[8] > 0
        assert rebound[10] == pytest.approx(0)

    def test_multiple_calls_rebound(self):
        """测试多次调用的反弹负荷叠加。"""
        n_slots = 30
        power_schedule = np.zeros(n_slots)
        power_schedule[2:4] = 100.0
        power_schedule[15:17] = 100.0

        rebound = compute_rebound_single_resource(
            power_schedule,
            rebound_factor=0.5,
            rebound_duration_min=30,
            slot_minutes=15
        )

        assert rebound[4] > 0
        assert rebound[17] > 0

    def test_find_call_end_slots(self):
        """测试调用结束时段识别。"""
        active_slots = np.array([2, 3, 4, 8, 9, 15])
        ends = find_call_end_slots(active_slots)
        assert ends == [4, 9, 15]

    def test_find_call_start_slots(self):
        """测试调用开始时段识别。"""
        active_slots = np.array([2, 3, 4, 8, 9, 15])
        starts = find_call_start_slots(active_slots)
        assert starts == [2, 8, 15]

    def test_no_calls_no_rebound(self):
        """没有调用时反弹为零。"""
        n_slots = 20
        power_schedule = np.zeros(n_slots)
        rebound = compute_rebound_single_resource(
            power_schedule, rebound_factor=0.5, rebound_duration_min=30
        )
        assert np.all(rebound == 0)

    def test_compute_net_load(self):
        """测试净负荷计算。"""
        baseline = np.array([100.0, 200.0, 300.0, 200.0, 100.0])
        schedule = np.array([[0, 50, 50, 0, 0]])
        register = pd.DataFrame([{
            'resource_id': 'R1',
            'rebound_factor': 1.0,
            'rebound_duration_min': 15
        }])

        net_load, rebound = compute_net_load(baseline, schedule, register)

        assert net_load[0] == pytest.approx(100.0)
        assert net_load[1] == pytest.approx(150.0)
        assert net_load[2] == pytest.approx(250.0)


class TestConstraints:
    """约束校验测试。"""

    def test_valid_schedule_passes(self):
        """合法的调度方案通过校验。"""
        register = _create_simple_register()
        n_slots = 96

        schedule = np.zeros((2, n_slots))
        schedule[0, 20:24] = 80.0
        schedule[1, 30:36] = 100.0

        valid, errors = validate_schedule(schedule, register)
        assert valid, f"期望通过校验，但错误: {errors}"
        assert len(errors) == 0

    def test_power_limit_violation(self):
        """功率超限能被检测到。"""
        register = _create_simple_register()
        n_slots = 96

        schedule = np.zeros((2, n_slots))
        schedule[0, 20] = 200.0

        valid, errors = validate_schedule(schedule, register)
        assert not valid
        assert any('超过上限' in e for e in errors)

    def test_max_calls_violation(self):
        """调用次数超限能被检测到。"""
        register = _create_simple_register()
        n_slots = 96

        schedule = np.zeros((2, n_slots))
        schedule[0, 10] = 50.0
        schedule[0, 20] = 50.0
        schedule[0, 30] = 50.0
        schedule[0, 40] = 50.0

        valid, errors = validate_schedule(schedule, register)
        assert not valid
        assert any('调用次数' in e for e in errors)

    def test_duration_violation(self):
        """单次时长超限能被检测到。"""
        register = _create_simple_register()
        n_slots = 96

        schedule = np.zeros((2, n_slots))
        schedule[0, 10:20] = 50.0

        valid, errors = validate_schedule(schedule, register)
        assert not valid
        assert any('持续' in e for e in errors)

    def test_rest_interval_violation(self):
        """休息间隔不足能被检测到。"""
        register = _create_simple_register()
        n_slots = 96

        schedule = np.zeros((2, n_slots))
        schedule[0, 10:12] = 50.0
        schedule[0, 13:15] = 50.0

        valid, errors = validate_schedule(schedule, register)
        assert not valid
        assert any('休息' in e for e in errors)

    def test_count_resource_calls(self):
        """测试调用次数统计。"""
        n_slots = 96
        schedule = np.zeros((1, n_slots))
        schedule[0, 10:12] = 50.0
        schedule[0, 20:22] = 50.0
        schedule[0, 35] = 50.0

        counts = count_resource_calls(schedule)
        assert counts[0] == 3

    def test_check_peak_constraint_pass(self):
        """测试峰值约束检查（通过）。"""
        load = np.array([100, 200, 150, 100])
        passes, slots, deficit = check_peak_constraint(load, target_peak=250)
        assert passes
        assert len(slots) == 0
        assert deficit == pytest.approx(0)

    def test_check_peak_constraint_fail(self):
        """测试峰值约束检查（不通过）。"""
        load = np.array([100, 200, 300, 150])
        passes, slots, deficit = check_peak_constraint(load, target_peak=200)
        assert not passes
        assert 2 in slots
        assert deficit == pytest.approx(100)


def _create_peak_baseline(n_slots=96):
    """创建一个钟形峰的基线负荷曲线。"""
    baseline = np.ones(n_slots) * 500.0

    peak_center = 50
    peak_width = 15
    for i in range(n_slots):
        baseline[i] += 300.0 * np.exp(-((i - peak_center) ** 2) / (2 * peak_width ** 2))

    return baseline


class TestSchedulerWithoutRebound:
    """不考虑反弹的调度引擎测试。"""

    def test_schedule_reduces_peak(self):
        """调度后峰值应该降低。"""
        baseline = _create_peak_baseline()
        register = _create_simple_register()
        target = 700.0

        result = greedy_schedule_without_rebound(baseline, target, register)

        assert result.peak_after_schedule < result.peak_before_schedule

    def test_all_constraints_satisfied(self):
        """调度方案应该满足所有约束。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 700.0

        result = greedy_schedule_without_rebound(baseline, target, register)

        valid, errors = validate_schedule(result.schedule_matrix, register)
        assert valid, f"约束校验失败: {errors}"

    def test_calls_count_matches(self):
        """统计的调用次数应该一致。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 650.0

        result = greedy_schedule_without_rebound(baseline, target, register)

        counts = count_resource_calls(result.schedule_matrix)
        for i, rid in enumerate(register['resource_id']):
            assert counts[i] == result.resource_calls[rid]

    def test_deterministic_result(self):
        """相同输入应该得到相同结果（确定性）。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 700.0

        result1 = greedy_schedule_without_rebound(baseline, target, register)
        result2 = greedy_schedule_without_rebound(baseline, target, register)

        assert np.array_equal(result1.schedule_matrix, result2.schedule_matrix)
        assert result1.total_cost == pytest.approx(result2.total_cost)

    def test_insufficient_resources_detected(self):
        """资源不足时应该报告无法达标。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 400.0

        result = greedy_schedule_without_rebound(baseline, target, register)

        assert not result.success
        assert len(result.deficit_slots) > 0
        assert result.deficit_amount > 0

    def test_sufficient_resources_meet_target(self):
        """资源充足时应该能达到目标。"""
        baseline = np.ones(96) * 500.0
        baseline[40:60] = 550.0

        register_data = [{
            'resource_id': 'R001',
            'name': '测试',
            'type': '空调',
            'max_power_kw': 100.0,
            'max_duration_min': 600,
            'max_calls_per_day': 10,
            'min_rest_min': 0,
            'cost_per_kw': 1.0,
            'rebound_factor': 0.0,
            'rebound_duration_min': 0
        }]
        register = pd.DataFrame(register_data)
        target = 520.0

        result = greedy_schedule_without_rebound(baseline, target, register)

        scheduled_peak = np.max(result.scheduled_load)
        assert scheduled_peak <= target + 1e-6, (
            f"调度后峰值 {scheduled_peak} 超过目标 {target}"
        )


class TestSchedulerWithRebound:
    """考虑反弹的调度引擎测试。"""

    def test_rebound_load_is_nonzero(self):
        """考虑反弹时应该有反弹负荷。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 700.0

        result = greedy_schedule_with_rebound(baseline, target, register)

        assert np.any(result.rebound_load > 0)

    def test_scheduled_load_includes_rebound(self):
        """调度后负荷应该包含反弹。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 700.0

        result = greedy_schedule_with_rebound(baseline, target, register)

        expected = (result.baseline_load
                    - result.schedule_matrix.sum(axis=0)
                    + result.rebound_load)
        assert np.allclose(result.scheduled_load, expected)

    def test_all_constraints_satisfied(self):
        """考虑反弹的调度也应该满足约束。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 700.0

        result = greedy_schedule_with_rebound(baseline, target, register)

        valid, errors = validate_schedule(result.schedule_matrix, register)
        assert valid, f"约束校验失败: {errors}"

    def test_deterministic_result(self):
        """考虑反弹的调度也应该是确定性的。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 700.0

        result1 = greedy_schedule_with_rebound(baseline, target, register)
        result2 = greedy_schedule_with_rebound(baseline, target, register)

        assert np.array_equal(result1.schedule_matrix, result2.schedule_matrix)
        assert result1.total_cost == pytest.approx(result2.total_cost)

    def test_insufficient_resources_detected(self):
        """资源不足时应该报告无法达标。"""
        baseline = _create_simple_baseline()
        register = _create_simple_register()
        target = 400.0

        result = greedy_schedule_with_rebound(baseline, target, register)

        assert not result.success
        assert len(result.deficit_slots) > 0

    def test_rebound_peak_is_higher(self):
        """考虑反弹后的峰值通常高于不考虑反弹的。"""
        baseline = _create_simple_baseline()

        register_data = [{
            'resource_id': 'R001',
            'name': '测试',
            'type': '空调',
            'max_power_kw': 200.0,
            'max_duration_min': 300,
            'max_calls_per_day': 5,
            'min_rest_min': 15,
            'cost_per_kw': 1.0,
            'rebound_factor': 0.8,
            'rebound_duration_min': 60
        }]
        register = pd.DataFrame(register_data)
        target = 650.0

        result_no_rebound = greedy_schedule_without_rebound(
            baseline, target, register
        )
        result_with_rebound = greedy_schedule_with_rebound(
            baseline, target, register
        )

        net_no_rebound = (baseline
                          - result_no_rebound.schedule_matrix.sum(axis=0)
                          + compute_rebound_all_resources(
                              result_no_rebound.schedule_matrix, register))
        peak_naive_with_rebound = np.max(net_no_rebound)

        assert result_with_rebound.peak_after_schedule <= peak_naive_with_rebound + 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
