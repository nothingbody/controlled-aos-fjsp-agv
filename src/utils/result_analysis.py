"""实验结果分析与表格生成

自动生成论文所需的LaTeX表格、箱线图、算子频率热力图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


# ===== LaTeX表格生成 =====

def generate_latex_table(
    df: pd.DataFrame,
    caption: str = "Experimental Results",
    label: str = "tab:results",
    highlight_best: bool = True,
    metric_cols: list = None,
    higher_better: dict = None,
) -> str:
    """将DataFrame转为LaTeX表格

    Args:
        df: 结果DataFrame
        caption: 表格标题
        label: LaTeX标签
        highlight_best: 是否加粗最优值
        metric_cols: 指标列名列表
        higher_better: {col_name: True/False}，指定越大越好还是越小越好

    Returns:
        LaTeX表格字符串
    """
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c not in ['instance', 'method', 'dataset']]
    if higher_better is None:
        higher_better = {c: ('HV' in c or 'NSol' in c) for c in metric_cols}

    # 找每列最优值
    if highlight_best and 'method' in df.columns:
        best_vals = {}
        for col in metric_cols:
            if higher_better.get(col, False):
                best_vals[col] = df.groupby('instance')[col].transform('max')
            else:
                best_vals[col] = df.groupby('instance')[col].transform('min')

    latex = df.to_latex(index=False, float_format="%.2f", escape=False)

    lines = [
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\small",
        latex,
        f"\\end{{table}}",
    ]
    return '\n'.join(lines)


def results_to_pivot(df: pd.DataFrame, value_col: str = 'HV_mean',
                     index_col: str = 'instance', columns_col: str = 'method') -> pd.DataFrame:
    """将长格式结果转为宽格式透视表"""
    return df.pivot_table(values=value_col, index=index_col,
                          columns=columns_col, aggfunc='mean')


def add_rank_column(df: pd.DataFrame, metric_col: str = 'HV_mean',
                    higher_better: bool = True) -> pd.DataFrame:
    """为每个实例添加方法排名列"""
    df = df.copy()
    df['rank'] = df.groupby('instance')[metric_col].rank(
        ascending=not higher_better, method='min'
    )
    return df


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成各方法的汇总统计表（平均排名、胜/平/负）"""
    if 'rank' not in df.columns:
        df = add_rank_column(df)

    summary = df.groupby('method').agg({
        'HV_mean': ['mean', 'std'],
        'Cmax_mean': 'mean',
        'TEC_mean': 'mean',
        'Time_mean': 'mean',
        'rank': 'mean',
    }).round(2)
    summary.columns = ['HV_avg', 'HV_std', 'Cmax_avg', 'TEC_avg', 'Time_avg', 'Avg_Rank']
    summary = summary.sort_values('Avg_Rank')
    return summary


# ===== 箱线图 =====

def plot_boxplot(
    results: Dict[str, List[float]],
    ylabel: str = "HV",
    title: str = "Performance Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    rotation: int = 30,
):
    """绘制各方法的性能箱线图

    Args:
        results: {method_name: [run1_value, run2_value, ...]}
    """
    fig, ax = plt.subplots(figsize=figsize)

    names = list(results.keys())
    data = [results[n] for n in names]

    bp = ax.boxplot(data, labels=names, patch_artist=True, notch=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=rotation, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===== 算子选择频率热力图 =====

def plot_operator_heatmap(
    operator_history: List[int],
    num_operators: int = 10,
    window_size: int = 50,
    operator_names: list = None,
    title: str = "Operator Selection Frequency by Evolution Stage",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
):
    """绘制算子选择频率热力图

    对应论文第7章 7.7.3节

    Args:
        operator_history: 每代选择的算子ID列表
        num_operators: 算子总数
        window_size: 滑动窗口大小（将进化过程分成若干阶段）
    """
    if operator_names is None:
        operator_names = ['POX', 'JBX', 'UnifMA', 'UnifAGV', '2Point',
                          'Swap', 'Insert', 'MachRe', 'AGVRe', 'SpeedAdj']

    history = np.array(operator_history)
    n_gens = len(history)
    n_windows = max(1, n_gens // window_size)

    freq_matrix = np.zeros((num_operators, n_windows))
    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n_gens)
        window_ops = history[start:end]
        for op in window_ops:
            if 0 <= op < num_operators:
                freq_matrix[op, w] += 1
        total = max(end - start, 1)
        freq_matrix[:, w] /= total

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(freq_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_yticks(range(num_operators))
    ax.set_yticklabels(operator_names[:num_operators], fontsize=10)
    ax.set_xlabel('Evolution Stage', fontsize=12)
    ax.set_ylabel('Operator', fontsize=12)
    ax.set_title(title, fontsize=14)

    stage_labels = [f'{w * window_size}-{min((w + 1) * window_size, n_gens)}'
                    for w in range(n_windows)]
    ax.set_xticks(range(n_windows))
    ax.set_xticklabels(stage_labels, rotation=45, ha='right', fontsize=8)

    plt.colorbar(im, ax=ax, label='Selection Frequency')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===== 动态响应时间线 =====

def plot_dynamic_timeline(
    response_log: List[dict],
    title: str = "Dynamic Event Response Timeline",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
):
    """绘制动态事件响应时间线

    对应论文第7章 7.7.5节

    Args:
        response_log: GradedRescheduler.response_log
    """
    if not response_log:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1],
                                    sharex=True)

    times = [r['time'] for r in response_log]
    disruptions = [r['disruption'] for r in response_log]
    levels = [r['level'] for r in response_log]
    event_types = [r['event_type'] for r in response_log]

    # 上图：扰动程度 + 阈值线
    level_colors = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}
    type_markers = {'new_job': 'o', 'machine_breakdown': 's', 'agv_breakdown': '^'}

    for i in range(len(times)):
        marker = type_markers.get(event_types[i], 'o')
        color = level_colors.get(levels[i], 'gray')
        ax1.scatter(times[i], disruptions[i], c=color, marker=marker,
                    s=80, edgecolors='black', linewidth=0.5, zorder=5)

    ax1.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='θ₁=0.15')
    ax1.axhline(y=0.40, color='red', linestyle='--', alpha=0.7, label='θ₂=0.40')
    ax1.set_ylabel('Disruption D', fontsize=11)
    ax1.set_title(title, fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 下图：Cmax变化
    cmax_before = [r['makespan_before'] for r in response_log]
    cmax_after = [r['makespan_after'] for r in response_log]

    ax2.bar(times, [a - b for a, b in zip(cmax_after, cmax_before)],
            width=max(1, (max(times) - min(times)) / len(times) / 2),
            color=['#e74c3c' if a > b else '#2ecc71'
                   for a, b in zip(cmax_after, cmax_before)],
            alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('ΔCmax', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ===== 参数敏感性图 =====

def plot_sensitivity(
    param_name: str,
    values: list,
    hvs: list,
    times: list = None,
    default_value: float = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """绘制单参数敏感性曲线

    对应论文第7章 7.6节
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    color_hv = '#2c3e50'
    ax1.plot(values, hvs, 'o-', color=color_hv, linewidth=2, markersize=8, label='HV')
    ax1.set_xlabel(param_name, fontsize=12)
    ax1.set_ylabel('HV', fontsize=12, color=color_hv)
    ax1.tick_params(axis='y', labelcolor=color_hv)

    if default_value is not None and default_value in values:
        idx = values.index(default_value)
        ax1.axvline(x=default_value, color='gray', linestyle=':', alpha=0.7)
        ax1.scatter([default_value], [hvs[idx]], s=150, c='red', zorder=10,
                    label=f'Default={default_value}')

    if times is not None:
        ax2 = ax1.twinx()
        color_time = '#e74c3c'
        ax2.plot(values, times, 's--', color=color_time, linewidth=1.5,
                 markersize=6, alpha=0.7, label='Time')
        ax2.set_ylabel('Time (s)', fontsize=12, color=color_time)
        ax2.tick_params(axis='y', labelcolor=color_time)

    ax1.set_title(f'Sensitivity Analysis: {param_name}', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
