"""收敛曲线可视化

对应论文第7章 7.7节
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def plot_convergence(
    histories: Dict[str, List[float]],
    title: str = "HV Convergence Curve",
    xlabel: str = "Generation",
    ylabel: str = "Hypervolume (HV)",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    log_scale: bool = False,
):
    """绘制多种方法的收敛曲线对比

    Args:
        histories: {method_name: [gen0_hv, gen1_hv, ...]}
        title: 图标题
        save_path: 保存路径（None则显示）
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']

    for idx, (name, hv_list) in enumerate(histories.items()):
        gens = list(range(len(hv_list)))
        # 每隔一定间隔标记点
        mark_every = max(1, len(hv_list) // 10)
        ax.plot(gens, hv_list, label=name,
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)],
                markevery=mark_every, markersize=5, linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multi_run_convergence(
    all_histories: Dict[str, List[List[float]]],
    title: str = "HV Convergence (mean ± std)",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """绘制多次运行的收敛曲线（含置信区间）

    Args:
        all_histories: {method_name: [[run1_hvs], [run2_hvs], ...]}
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_histories), 1)))

    for idx, (name, runs) in enumerate(all_histories.items()):
        # 对齐长度
        min_len = min(len(r) for r in runs)
        aligned = np.array([r[:min_len] for r in runs])

        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)
        gens = np.arange(min_len)

        color = colors[idx % len(colors)]
        ax.plot(gens, mean, label=name, color=color, linewidth=2)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Hypervolume (HV)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
