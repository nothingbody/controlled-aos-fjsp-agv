"""Pareto前沿可视化

对应论文第7章 7.7.2节
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_pareto_2d(
    fronts: Dict[str, np.ndarray],
    obj_x: int = 0, obj_y: int = 1,
    xlabel: str = "Makespan (Cmax)",
    ylabel: str = "Total Energy (TEC)",
    title: str = "Pareto Front Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 7),
):
    """绘制多种方法的2D Pareto前沿对比

    Args:
        fronts: {method_name: objectives_array (N, M)}
        obj_x: x轴目标索引
        obj_y: y轴目标索引
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fronts), 1)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', '+']

    for idx, (name, objs) in enumerate(fronts.items()):
        if len(objs) == 0:
            continue
        # 按x轴排序以便连线
        sorted_idx = np.argsort(objs[:, obj_x])
        x = objs[sorted_idx, obj_x]
        y = objs[sorted_idx, obj_y]

        ax.scatter(x, y, label=name,
                   color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)],
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        # 连接Pareto前沿
        ax.plot(x, y, color=colors[idx % len(colors)],
                alpha=0.4, linewidth=1)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pareto_3d(
    fronts: Dict[str, np.ndarray],
    labels: tuple = ("Cmax", "TEC", "Workload Balance"),
    title: str = "3D Pareto Front",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
):
    """绘制3D Pareto前沿"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fronts), 1)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, (name, objs) in enumerate(fronts.items()):
        if len(objs) == 0 or objs.shape[1] < 3:
            continue
        ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2],
                   label=name,
                   color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)],
                   s=40, alpha=0.7)

    ax.set_xlabel(labels[0], fontsize=10)
    ax.set_ylabel(labels[1], fontsize=10)
    ax.set_zlabel(labels[2], fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_parallel_coordinates(
    objs: np.ndarray,
    labels: list = None,
    title: str = "Parallel Coordinates of Pareto Solutions",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """平行坐标图展示Pareto解集

    适用于4目标或更多目标时无法用散点图展示的情况
    """
    if labels is None:
        labels = [f"f{i+1}" for i in range(objs.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)

    # 归一化到[0,1]
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-10] = 1e-10
    normalized = (objs - mins) / ranges

    x = np.arange(len(labels))
    cmap = plt.cm.viridis

    for i in range(len(normalized)):
        color = cmap(i / max(len(normalized) - 1, 1))
        ax.plot(x, normalized[i], color=color, alpha=0.5, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
