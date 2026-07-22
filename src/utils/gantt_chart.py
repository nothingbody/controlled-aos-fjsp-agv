"""甘特图可视化"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_gantt(schedule, instance, title="FJSP-AGV Schedule", save_path=None):
    """绘制调度甘特图

    Args:
        schedule: Schedule对象
        instance: FJSPAGVInstance对象
        title: 图标题
        save_path: 保存路径（None则显示）
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])

    colors = plt.cm.tab20(np.linspace(0, 1, max(instance.num_jobs, 1)))

    # 上图：机器甘特图
    ax1 = axes[0]
    for u in range(instance.num_machines):
        ops = schedule.machine_schedule.get(u, [])
        for (i, j, start, end) in ops:
            ax1.barh(u, end - start, left=start, height=0.6,
                     color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            if end - start > 2:
                ax1.text((start + end) / 2, u, f'J{i+1}O{j+1}',
                         ha='center', va='center', fontsize=6)

    ax1.set_yticks(range(instance.num_machines))
    ax1.set_yticklabels([f'M{u+1}' for u in range(instance.num_machines)])
    ax1.set_xlabel('Time')
    ax1.set_title(f'{title} - Machine Schedule')
    ax1.invert_yaxis()

    # 下图：AGV甘特图
    ax2 = axes[1]
    for l in range(instance.num_agv):
        tasks = schedule.agv_schedule.get(l, [])
        for (i, j, start, end) in tasks:
            ax2.barh(l, end - start, left=start, height=0.6,
                     color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)

    ax2.set_yticks(range(instance.num_agv))
    ax2.set_yticklabels([f'AGV{l+1}' for l in range(instance.num_agv)])
    ax2.set_xlabel('Time')
    ax2.set_title('AGV Schedule')
    ax2.invert_yaxis()

    # 图例
    patches = [mpatches.Patch(color=colors[i % len(colors)], label=f'Job {i+1}')
               for i in range(min(instance.num_jobs, 20))]
    fig.legend(handles=patches, loc='center right', fontsize=7,
               bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
