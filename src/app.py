import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import html
from src.data.generator import (
    generate_baseline_load, generate_resource_register,
    validate_baseline_data, validate_register_data,
    clean_register_data
)
from src.dashboard.layout import create_layout
from src.dashboard.callbacks import register_callbacks


def main():
    """主函数：启动 Dash 应用。"""
    baseline_df = generate_baseline_load(seed=42)
    register_df = generate_resource_register(seed=42)

    baseline_errors = validate_baseline_data(baseline_df)
    register_errors = validate_register_data(register_df)

    if baseline_errors:
        print("基线数据警告:")
        for e in baseline_errors:
            print(f"  - {e}")

    if register_errors:
        print("资源台账警告:")
        for e in register_errors:
            print(f"  - {e}")

    register_df = clean_register_data(register_df)

    peak_load = baseline_df['load_kw'].max()
    initial_target = peak_load * 0.88

    app = dash.Dash(__name__)
    app.title = '柔性负荷削峰调度分析看板'

    app.layout = create_layout(baseline_df, register_df, initial_target)

    register_callbacks(app, baseline_df, register_df)

    print("看板启动中...")
    print(f"基线峰值: {peak_load:.1f} kW")
    print(f"初始目标峰值: {initial_target:.1f} kW")
    print(f"资源数量: {len(register_df)} 个")
    print(f"访问地址: http://localhost:7913")

    app.run_server(host='0.0.0.0', port=7913, debug=False)


if __name__ == '__main__':
    main()
