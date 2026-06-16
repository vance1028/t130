import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context
from src.scheduler.engine import (
    greedy_schedule_with_rebound,
    greedy_schedule_without_rebound
)
from src.metrics.calculator import compute_schedule_metrics, compare_schedules
from src.rebound.model import compute_rebound_all_resources
from src.constraints.validator import count_resource_calls


def register_callbacks(app, baseline_df, register_df):
    """注册所有 Dash 回调。

    Args:
        app: Dash 应用实例
        baseline_df: 基线负荷数据
        register_df: 资源台账
    """

    baseline_load = baseline_df['load_kw'].values
    times = baseline_df['time'].values
    resource_ids = register_df['resource_id'].tolist()
    n_slots = len(baseline_load)

    @app.callback(
        [Output('target-peak-slider', 'value'),
         Output('target-peak-input', 'value')],
        [Input('target-peak-slider', 'value'),
         Input('target-peak-input', 'value')]
    )
    def sync_target_peak(slider_val, input_val):
        """同步滑块和输入框的值。"""
        ctx = callback_context
        if not ctx.triggered:
            return slider_val, input_val

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'target-peak-slider':
            return slider_val, slider_val
        else:
            if input_val is None:
                return slider_val, slider_val
            return input_val, input_val

    @app.callback(
        [Output('load-curve-graph', 'figure'),
         Output('schedule-gantt-graph', 'figure'),
         Output('stat-peak-before', 'children'),
         Output('stat-peak-after', 'children'),
         Output('stat-curtailment', 'children'),
         Output('stat-cost', 'children'),
         Output('stat-meets-target', 'children'),
         Output('stat-meets-target', 'style'),
         Output('stat-rebound', 'children'),
         Output('warning-message', 'children'),
         Output('register-table', 'data'),
         Output('compare-no-rebound-peak', 'children'),
         Output('compare-no-rebound-cost', 'children'),
         Output('compare-with-rebound-peak', 'children'),
         Output('compare-with-rebound-cost', 'children'),
         Output('compare-diff-peak', 'children')],
        [Input('target-peak-slider', 'value'),
         Input('highlight-resource-dropdown', 'value')],
        [State('target-peak-input', 'value')]
    )
    def update_dashboard(target_peak, highlight_resource, input_val):
        """更新整个看板的所有内容。"""
        if target_peak is None:
            target_peak = input_val
        if target_peak is None:
            target_peak = 1000.0

        result_rebound = greedy_schedule_with_rebound(
            baseline_load, target_peak, register_df
        )
        result_no_rebound = greedy_schedule_without_rebound(
            baseline_load, target_peak, register_df
        )

        metrics = compute_schedule_metrics(
            baseline_load, result_rebound.schedule_matrix,
            register_df, target_peak
        )

        comparison = compare_schedules(
            baseline_load,
            result_no_rebound.schedule_matrix,
            result_rebound.schedule_matrix,
            register_df,
            target_peak
        )

        load_fig = create_load_curve_figure(
            times, baseline_load, result_rebound, target_peak, highlight_resource
        )

        gantt_fig = create_schedule_gantt_figure(
            result_rebound.schedule_matrix, register_df,
            times, highlight_resource
        )

        table_data = update_register_table_data(
            register_df, result_rebound.schedule_matrix
        )

        peak_before_str = f"{metrics['peak_before_kw']:.1f} kW"
        peak_after_str = f"{metrics['peak_after_kw']:.1f} kW"
        curtailment_str = f"{metrics['total_curtailment_kwh']:.1f} kWh"
        cost_str = f"{metrics['total_cost']:.2f} 元"
        rebound_str = f"{metrics['total_rebound_kwh']:.1f} kWh"

        if metrics['meets_target']:
            meets_str = '✓ 达标'
            meets_style = {'fontSize': '28px', 'fontWeight': 'bold',
                           'color': '#27ae60'}
        else:
            meets_str = '✗ 未达标'
            meets_style = {'fontSize': '28px', 'fontWeight': 'bold',
                           'color': '#e74c3c'}

        warning = ''
        if not metrics['meets_target']:
            warning = (f"警告：无法达标！有 {metrics['over_target_slots']} 个时段"
                       f"超出目标线，总缺口 {metrics['total_deficit_kwh']:.2f} kWh")

        no_rebound_peak = f"{comparison['no_rebound_simulation']['peak_after_schedule_kw']:.1f} kW"
        no_rebound_cost = f"成本: {comparison['no_rebound_simulation']['total_cost']:.2f} 元"
        with_rebound_peak = f"{comparison['with_rebound_simulation']['peak_after_schedule_kw']:.1f} kW"
        with_rebound_cost = f"成本: {comparison['with_rebound_simulation']['total_cost']:.2f} 元"

        diff_peak_kw = comparison['peak_difference_kw']
        if diff_peak_kw > 0:
            diff_str = f"考虑反弹后峰值高 {diff_peak_kw:.1f} kW"
        else:
            diff_str = f"考虑反弹后峰值低 {-diff_peak_kw:.1f} kW"

        return (
            load_fig, gantt_fig,
            peak_before_str, peak_after_str, curtailment_str,
            cost_str, meets_str, meets_style, rebound_str,
            warning, table_data,
            no_rebound_peak, no_rebound_cost,
            with_rebound_peak, with_rebound_cost,
            diff_str
        )


def create_load_curve_figure(times, baseline_load, result, target_peak,
                             highlight_resource='all'):
    """创建负荷曲线图。

    Args:
        times: 时间数组
        baseline_load: 基线负荷
        result: 调度结果
        target_peak: 目标峰值
        highlight_resource: 高亮的资源ID，'all'表示全部

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times,
        y=baseline_load,
        mode='lines',
        name='基线预测',
        line=dict(color='#3498db', width=2),
        fill=None
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=result.scheduled_load,
        mode='lines',
        name='调度后实际负荷',
        line=dict(color='#27ae60', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=result.rebound_load,
        mode='lines',
        name='反弹负荷',
        line=dict(color='#e74c3c', width=1, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))

    fig.add_hline(
        y=target_peak,
        line_dash='dash',
        line_color='#e67e22',
        line_width=2,
        annotation_text=f'目标峰值: {target_peak:.0f} kW',
        annotation_position='top right'
    )

    over_mask = result.scheduled_load > target_peak
    if over_mask.any():
        fig.add_trace(go.Scatter(
            x=times[over_mask],
            y=result.scheduled_load[over_mask],
            mode='markers',
            name='超限点',
            marker=dict(color='red', size=8, symbol='circle')
        ))

    fig.update_layout(
        title='全天负荷曲线对比',
        xaxis_title='时间',
        yaxis_title='功率 (kW)',
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.15),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


def create_schedule_gantt_figure(schedule_matrix, register_df, times,
                                 highlight_resource='all'):
    """创建资源调度甘特图（热力图形式）。

    Args:
        schedule_matrix: 调度矩阵
        register_df: 资源台账
        times: 时间数组
        highlight_resource: 高亮资源

    Returns:
        Plotly Figure
    """
    n_resources, n_slots = schedule_matrix.shape

    fig = go.Figure()

    colorscale = [
        [0, 'rgba(200, 200, 200, 0.3)'],
        [0.5, 'rgba(52, 152, 219, 0.6)'],
        [1, 'rgba(39, 174, 96, 1.0)']
    ]

    z_data = []
    y_labels = []

    for r_idx in range(n_resources):
        row = register_df.iloc[r_idx]
        y_labels.append(f"{row['name']} ({row['resource_id']})")
        z_data.append(schedule_matrix[r_idx, :].tolist())

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[pd.Timestamp(t).strftime('%H:%M') for t in times],
        y=y_labels,
        colorscale=colorscale,
        colorbar=dict(title='削减功率 (kW)'),
        hoverongaps=False,
        hovertemplate='时段: %{x}<br>资源: %{y}<br>削减: %{z:.1f} kW<extra></extra>'
    ))

    fig.update_layout(
        title='各时段资源调用情况',
        xaxis_title='时间',
        yaxis_title='资源',
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig


def update_register_table_data(register_df, schedule_matrix):
    """更新台账表格数据，添加实际调用次数和是否触顶。"""
    call_counts = count_resource_calls(schedule_matrix)

    data = register_df.copy()
    data['actual_calls'] = call_counts
    data['is_maxed_out'] = ['是' if c >= m else '否'
                            for c, m in zip(call_counts, data['max_calls_per_day'])]

    return data.to_dict('records')
