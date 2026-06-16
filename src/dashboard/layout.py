import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_layout(baseline_df: pd.DataFrame, register_df: pd.DataFrame,
                  initial_target_peak: float) -> html.Div:
    """创建看板布局。

    Args:
        baseline_df: 基线负荷数据
        register_df: 资源台账
        initial_target_peak: 初始目标峰值

    Returns:
        Dash 布局组件
    """
    peak_value = baseline_df['load_kw'].max()
    min_target = peak_value * 0.7
    max_target = peak_value * 1.0

    return html.Div([
        html.H1('柔性负荷削峰调度分析看板',
                style={'textAlign': 'center', 'marginBottom': '20px',
                       'color': '#2c3e50'}),

        html.Div([
            html.Div([
                html.Label('目标峰值 (kW):', style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='target-peak-slider',
                    min=min_target,
                    max=max_target,
                    value=initial_target_peak,
                    step=10,
                    marks={int(v): f'{int(v)}'
                           for v in np.linspace(min_target, max_target, 8)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '70%', 'display': 'inline-block'}),
            html.Div([
                html.Label('或直接输入:', style={'fontWeight': 'bold'}),
                dcc.Input(
                    id='target-peak-input',
                    type='number',
                    value=initial_target_peak,
                    min=min_target,
                    max=max_target,
                    step=10,
                    style={'width': '120px', 'marginLeft': '10px',
                           'padding': '5px'}
                )
            ], style={'width': '28%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'textAlign': 'right'})
        ], style={'marginBottom': '20px', 'padding': '15px',
                  'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),

        html.Div([
            html.H3('负荷曲线对比', style={'marginTop': '0'}),
            dcc.Graph(id='load-curve-graph', style={'height': '400px'}),
            html.Div(id='warning-message', style={'color': '#e74c3c',
                                                  'fontWeight': 'bold',
                                                  'marginTop': '10px'})
        ], style={'padding': '15px', 'backgroundColor': 'white',
                  'borderRadius': '8px', 'marginBottom': '20px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

        html.Div([
            html.Div([
                html.H4('调度前峰值', style={'color': '#7f8c8d'}),
                html.Div(id='stat-peak-before',
                         style={'fontSize': '28px', 'fontWeight': 'bold',
                                'color': '#3498db'})
            ], className='stat-card', style=stat_card_style()),
            html.Div([
                html.H4('调度后峰值', style={'color': '#7f8c8d'}),
                html.Div(id='stat-peak-after',
                         style={'fontSize': '28px', 'fontWeight': 'bold',
                                'color': '#27ae60'})
            ], className='stat-card', style=stat_card_style()),
            html.Div([
                html.H4('削峰量', style={'color': '#7f8c8d'}),
                html.Div(id='stat-curtailment',
                         style={'fontSize': '28px', 'fontWeight': 'bold',
                                'color': '#e67e22'})
            ], className='stat-card', style=stat_card_style()),
            html.Div([
                html.H4('总削减成本', style={'color': '#7f8c8d'}),
                html.Div(id='stat-cost',
                         style={'fontSize': '28px', 'fontWeight': 'bold',
                                'color': '#9b59b6'})
            ], className='stat-card', style=stat_card_style()),
            html.Div([
                html.H4('是否达标', style={'color': '#7f8c8d'}),
                html.Div(id='stat-meets-target',
                         style={'fontSize': '28px', 'fontWeight': 'bold'})
            ], className='stat-card', style=stat_card_style()),
            html.Div([
                html.H4('反弹新增负荷', style={'color': '#7f8c8d'}),
                html.Div(id='stat-rebound',
                         style={'fontSize': '28px', 'fontWeight': 'bold',
                                'color': '#e74c3c'})
            ], className='stat-card', style=stat_card_style()),
        ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '20px',
                  'flexWrap': 'wrap'}),

        html.Div([
            html.H3('资源调度详情', style={'marginTop': '0'}),

            html.Div([
                html.Div([
                    html.Label('高亮资源:'),
                    dcc.Dropdown(
                        id='highlight-resource-dropdown',
                        options=[{'label': '全部', 'value': 'all'}] +
                                [{'label': f"{row['name']} ({row['resource_id']})",
                                  'value': row['resource_id']}
                                 for _, row in register_df.iterrows()],
                        value='all',
                        clearable=False,
                        style={'width': '250px', 'display': 'inline-block',
                               'marginLeft': '10px'}
                    ),
                ], style={'marginBottom': '10px'}),

                dcc.Graph(id='schedule-gantt-graph', style={'height': '350px'}),
            ]),

            html.Div([
                html.H4('资源台账', style={'marginTop': '20px'}),
                dash_table.DataTable(
                    id='register-table',
                    columns=[
                        {'name': '资源ID', 'id': 'resource_id'},
                        {'name': '名称', 'id': 'name'},
                        {'name': '类型', 'id': 'type'},
                        {'name': '功率上限(kW)', 'id': 'max_power_kw'},
                        {'name': '最长持续(分)', 'id': 'max_duration_min'},
                        {'name': '日调用上限', 'id': 'max_calls_per_day'},
                        {'name': '休息间隔(分)', 'id': 'min_rest_min'},
                        {'name': '单位成本(元/kW)', 'id': 'cost_per_kw'},
                        {'name': '反弹系数', 'id': 'rebound_factor'},
                        {'name': '反弹时长(分)', 'id': 'rebound_duration_min'},
                        {'name': '实际调用次数', 'id': 'actual_calls'},
                        {'name': '是否触顶', 'id': 'is_maxed_out'},
                    ],
                    data=register_df.to_dict('records'),
                    sort_action='native',
                    sort_mode='single',
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#f8f9fa',
                                  'fontWeight': 'bold'},
                    style_cell={'padding': '8px', 'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'column_id': 'is_maxed_out',
                                'filter_query': '{is_maxed_out} = "是"'},
                         'color': 'red', 'fontWeight': 'bold'},
                    ]
                )
            ])
        ], style={'padding': '15px', 'backgroundColor': 'white',
                  'borderRadius': '8px', 'marginBottom': '20px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

        html.Div([
            html.H3('方案对比：考虑反弹 vs 不考虑反弹', style={'marginTop': '0'}),
            html.Div([
                html.Div([
                    html.H5('不考虑反弹（贪心）',
                            style={'textAlign': 'center', 'color': '#e74c3c'}),
                    html.Div(id='compare-no-rebound-peak',
                             style={'fontSize': '24px', 'fontWeight': 'bold',
                                    'textAlign': 'center'}),
                    html.Div('模拟峰值(不考虑反弹)',
                             style={'textAlign': 'center', 'color': '#7f8c8d'}),
                    html.Div(id='compare-no-rebound-cost',
                             style={'fontSize': '18px', 'marginTop': '10px',
                                    'textAlign': 'center'}),
                ], style={'flex': '1', 'padding': '15px',
                          'backgroundColor': '#fdf0ef',
                          'borderRadius': '8px'}),

                html.Div([
                    html.H4('→', style={'textAlign': 'center',
                                        'lineHeight': '100px',
                                        'color': '#7f8c8d'}),
                    html.Div(id='compare-diff-peak',
                             style={'textAlign': 'center', 'fontSize': '14px',
                                    'color': '#e74c3c'}),
                ], style={'flex': '0 0 80px', 'display': 'flex',
                          'flexDirection': 'column', 'justifyContent': 'center'}),

                html.Div([
                    html.H5('考虑反弹（优化）',
                            style={'textAlign': 'center', 'color': '#27ae60'}),
                    html.Div(id='compare-with-rebound-peak',
                             style={'fontSize': '24px', 'fontWeight': 'bold',
                                    'textAlign': 'center'}),
                    html.Div('实际峰值(考虑反弹)',
                             style={'textAlign': 'center', 'color': '#7f8c8d'}),
                    html.Div(id='compare-with-rebound-cost',
                             style={'fontSize': '18px', 'marginTop': '10px',
                                    'textAlign': 'center'}),
                ], style={'flex': '1', 'padding': '15px',
                          'backgroundColor': '#eafaf1',
                          'borderRadius': '8px'}),
            ], style={'display': 'flex', 'gap': '20px'})
        ], style={'padding': '15px', 'backgroundColor': 'white',
                  'borderRadius': '8px', 'marginBottom': '20px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),

        dcc.Store(id='schedule-results-store')

    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1',
              'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})


def stat_card_style() -> dict:
    """统计卡片的样式。"""
    return {
        'flex': '1',
        'minWidth': '150px',
        'padding': '15px',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'textAlign': 'center',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }
