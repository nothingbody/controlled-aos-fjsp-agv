"""AOS升维分析图表：阶段划分 + 奖励相关性 + 算子分布变化"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

FIG_DIR = Path("results/figures/publication_v4")
FIG_DIR.mkdir(parents=True, exist_ok=True)

WONG = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'pink': '#CC79A7', 'red': '#D55E00', 'skyblue': '#56B4E9',
    'yellow': '#F0E442', 'black': '#000000',
}

def setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8, 'mathtext.fontset': 'stix',
        'axes.linewidth': 0.5, 'axes.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 6, 'legend.frameon': True,
        'figure.dpi': 150, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
    })

def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", format='pdf')
    fig.savefig(FIG_DIR / f"{name}.png", format='png')
    plt.close(fig)
    print(f"  [OK] {name}")


def plot_phase_transition_and_hv(data, inst_name='Mk06.fjs'):
    """Fig A: 阶段划分 + HV收敛曲线（叠加显示UCB→PPO过渡点）"""
    print(f"\n[Fig A] Phase transition + HV convergence ({inst_name})...")
    d = data[inst_name]
    hv = d['hv_history']
    switch = d['switch_gen']
    phases = d['phase_list']
    gens = list(range(len(hv)))

    fig, ax1 = plt.subplots(figsize=(7.48, 2.8))

    # HV曲线
    ax1.plot(gens, hv, color=WONG['blue'], linewidth=1.2, label='HV')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Hypervolume', color=WONG['blue'])
    ax1.tick_params(axis='y', labelcolor=WONG['blue'])

    # 标注阶段
    if switch > 0:
        ax1.axvline(switch, color=WONG['red'], linestyle='--', linewidth=1.0, alpha=0.8)
        ax1.axvspan(0, switch, alpha=0.08, color=WONG['orange'], label='Phase 1: UCB Exploration')
        ax1.axvspan(switch, len(hv), alpha=0.08, color=WONG['green'], label='Phase 2: PPO Exploitation')

        # 标注文字
        y_pos = max(hv) * 0.95
        ax1.annotate(f'$T_c = {switch}$', xy=(switch, y_pos),
                     xytext=(switch + 8, y_pos * 1.02),
                     fontsize=8, fontweight='bold', color=WONG['red'],
                     arrowprops=dict(arrowstyle='->', color=WONG['red'], lw=1.0))
        ax1.text(switch / 2, max(hv) * 0.15, 'Exploration\n(UCB)',
                 ha='center', fontsize=7, fontstyle='italic', color=WONG['orange'])
        ax1.text((switch + len(hv)) / 2, max(hv) * 0.15, 'Exploitation\n(PPO)',
                 ha='center', fontsize=7, fontstyle='italic', color=WONG['green'])

    ax1.legend(loc='lower right', fontsize=6)
    ax1.set_title(f'(a) Dual-Layer AOS Phase Transition on {inst_name.replace(".fjs","")}', fontsize=8)
    ax1.yaxis.grid(True, linewidth=0.3, alpha=0.2)
    ax1.set_axisbelow(True)

    save_fig(fig, 'fig_aos_phase_transition')


def plot_operator_distribution(data, inst_name='Mk06.fjs'):
    """Fig B: 算子使用分布随进化阶段变化（滑动窗口热力图）"""
    print(f"\n[Fig B] Operator distribution over generations ({inst_name})...")
    d = data[inst_name]
    phase_history = d['phase_history']  # [(gen, phase, op_id)]
    op_names = d['op_names']
    switch = d['switch_gen']
    n_ops = len(op_names)
    n_gens = len(phase_history)

    # 滑动窗口统计（窗口=10代）
    window = 10
    n_bins = n_gens // window
    freq_matrix = np.zeros((n_ops, n_bins))

    for bin_idx in range(n_bins):
        start = bin_idx * window
        end = min(start + window, n_gens)
        for gen, phase, op_id in phase_history[start:end]:
            freq_matrix[op_id, bin_idx] += 1
        total = freq_matrix[:, bin_idx].sum()
        if total > 0:
            freq_matrix[:, bin_idx] /= total

    fig, ax = plt.subplots(figsize=(7.48, 3.0))
    im = ax.imshow(freq_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.5,
                   interpolation='nearest')

    ax.set_yticks(range(n_ops))
    short_names = ['POX', 'JBX', 'UniMA', 'UniAGV', '2Pt',
                   'Swap', 'Insert', 'MaRe', 'AGVRe', 'SpdAdj']
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel(f'Generation (bins of {window})')
    ax.set_ylabel('Operator')
    xtick_pos = list(range(0, n_bins, max(1, n_bins // 10)))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([str(p * window) for p in xtick_pos], fontsize=6)

    # 标注阶段过渡
    if switch > 0:
        switch_bin = switch // window
        ax.axvline(switch_bin, color='white', linestyle='--', linewidth=1.5)
        ax.text(switch_bin - 0.5, -0.8, 'UCB', fontsize=7, color=WONG['orange'],
                fontweight='bold', ha='right')
        ax.text(switch_bin + 0.5, -0.8, 'PPO', fontsize=7, color=WONG['green'],
                fontweight='bold', ha='left')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Selection Frequency', fontsize=7)
    ax.set_title(f'(b) Operator Selection Distribution over Evolution ({inst_name.replace(".fjs","")})',
                 fontsize=8)

    save_fig(fig, 'fig_operator_distribution')


def plot_reward_hv_correlation(data, inst_name='Mk06.fjs'):
    """Fig C: 复合奖励 vs HV改善的相关性分析"""
    print(f"\n[Fig C] Reward vs HV correlation ({inst_name})...")
    d = data[inst_name]
    ucb_rewards = d['ucb_rewards']  # [(op_id, reward)]
    hv = d['hv_history']

    # 每代的奖励和HV改善
    rewards = [r for _, r in ucb_rewards[:len(hv)]]
    hv_changes = [0] + [hv[i] - hv[i-1] for i in range(1, len(hv))]
    hv_changes_norm = [c / max(abs(hv[max(0, i-1)]), 1e-10) for i, c in enumerate(hv_changes)]

    min_len = min(len(rewards), len(hv_changes_norm))
    rewards = rewards[:min_len]
    hv_changes_norm = hv_changes_norm[:min_len]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.48, 2.5))

    # 左图：散点图 + 回归线
    ax1.scatter(rewards, hv_changes_norm, s=8, alpha=0.5, color=WONG['blue'], edgecolors='none')
    if len(rewards) > 2:
        z = np.polyfit(rewards, hv_changes_norm, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(rewards), max(rewards), 100)
        ax1.plot(x_line, p(x_line), color=WONG['red'], linewidth=1.2, linestyle='--')

        # Pearson相关系数
        from scipy import stats
        r, pval = stats.pearsonr(rewards, hv_changes_norm)
        ax1.text(0.05, 0.92, f'Pearson r = {r:.3f}\np = {pval:.3e}',
                 transform=ax1.transAxes, fontsize=7,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Composite Reward')
    ax1.set_ylabel('Normalized HV Change')
    ax1.set_title('(c) Reward–HV Correlation', fontsize=8)
    ax1.yaxis.grid(True, linewidth=0.3, alpha=0.2)
    ax1.set_axisbelow(True)

    # 右图：奖励随代数变化（分阶段着色）
    switch = d['switch_gen']
    gens = list(range(min_len))

    if switch > 0 and switch < min_len:
        ax2.plot(gens[:switch], rewards[:switch], color=WONG['orange'],
                 linewidth=0.6, alpha=0.7, label='UCB phase')
        ax2.plot(gens[switch:], rewards[switch:], color=WONG['green'],
                 linewidth=0.6, alpha=0.7, label='PPO phase')
        ax2.axvline(switch, color=WONG['red'], linestyle='--', linewidth=0.8)
    else:
        ax2.plot(gens, rewards, color=WONG['blue'], linewidth=0.6, alpha=0.7)

    # 滑动均值
    if len(rewards) > 10:
        window = 10
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(rewards)), avg, color=WONG['black'],
                 linewidth=1.5, label='Moving avg (w=10)')

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Composite Reward')
    ax2.set_title('(d) Reward Trend by Phase', fontsize=8)
    ax2.legend(loc='lower right', fontsize=6)
    ax2.yaxis.grid(True, linewidth=0.3, alpha=0.2)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, 'fig_reward_analysis')


if __name__ == '__main__':
    setup_style()
    print("=" * 60)
    print("AOS Analysis Figures (Theory Support)")
    print("=" * 60)

    with open('results/aos_analysis_data.pkl', 'rb') as f:
        data = pickle.load(f)

    plot_phase_transition_and_hv(data, 'Mk06.fjs')
    plot_operator_distribution(data, 'Mk06.fjs')
    plot_reward_hv_correlation(data, 'Mk06.fjs')

    print(f"\nAll figures saved to: {FIG_DIR}")
