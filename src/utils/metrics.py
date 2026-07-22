"""多目标评价指标 + 统计检验

对应论文第7章 7.1.4节 和 7.1.5节
"""

import numpy as np
from scipy import stats
from src.algorithm.nsga3.selection import compute_hypervolume


def compute_igd(pareto_front: np.ndarray, reference_front: np.ndarray) -> float:
    """计算反转世代距离 (IGD)

    Args:
        pareto_front: 算法得到的Pareto前沿 (N, M)
        reference_front: 真实/参考Pareto前沿 (N_ref, M)

    Returns:
        IGD值（越小越好）
    """
    if len(pareto_front) == 0:
        return float('inf')

    distances = []
    for ref in reference_front:
        min_dist = np.min(np.linalg.norm(pareto_front - ref, axis=1))
        distances.append(min_dist)

    return np.mean(distances)


def compute_spread(pareto_front: np.ndarray) -> float:
    """计算分布指标 (Spread)

    Args:
        pareto_front: Pareto前沿 (N, M)

    Returns:
        Spread值（越小越好）
    """
    pareto_front = np.asarray(pareto_front, dtype=float)
    if pareto_front.size == 0:
        return 0.0
    if pareto_front.ndim != 2:
        raise ValueError("pareto_front must be a two-dimensional array")
    if not np.all(np.isfinite(pareto_front)):
        raise ValueError("pareto_front must contain only finite values")

    # Exact duplicate objective vectors are not distinct Pareto solutions and
    # must not add zero-length gaps.  Normalize each objective so that the
    # result is invariant to units (e.g., seconds versus joules).
    unique_pf = np.unique(pareto_front, axis=0)
    if len(unique_pf) <= 2:
        return 0.0

    minima = unique_pf.min(axis=0)
    ranges = unique_pf.max(axis=0) - minima
    scales = np.where(ranges > 1e-12, ranges, 1.0)
    normalized_pf = (unique_pf - minima) / scales

    sorted_pf = normalized_pf[np.argsort(normalized_pf[:, 0], kind="stable")]
    distances = np.linalg.norm(np.diff(sorted_pf, axis=0), axis=1)

    if len(distances) == 0:
        return 0.0

    mean_d = np.mean(distances)
    if mean_d < 1e-10:
        return 0.0

    spread = np.sum(np.abs(distances - mean_d)) / ((len(distances)) * mean_d)
    return spread


def compute_hv(pareto_front: np.ndarray, ref_point: np.ndarray = None) -> float:
    """计算超体积

    Args:
        pareto_front: Pareto前沿 (N, M)
        ref_point: 参考点（若None则自动设为各目标最大值*1.1）

    Returns:
        HV值（越大越好）
    """
    if len(pareto_front) == 0:
        return 0.0

    if ref_point is None:
        ref_point = pareto_front.max(axis=0) * 1.1

    return compute_hypervolume(pareto_front, ref_point)


# ===== 统计检验 =====

def wilcoxon_test(data_a: list, data_b: list, alpha: float = 0.05) -> dict:
    """Wilcoxon秩和检验（两组独立样本）

    对应论文7.1.5节：判断两种方法的性能差异是否显著

    Args:
        data_a: 方法A的多次运行结果（如30次HV值）
        data_b: 方法B的多次运行结果
        alpha: 显著性水平

    Returns:
        {'statistic': U值, 'p_value': p值, 'significant': 是否显著,
         'winner': 'A'/'B'/'tie', 'symbol': '+'/'-'/'≈'}
    """
    a, b = np.array(data_a), np.array(data_b)

    if len(a) < 2 or len(b) < 2:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False,
                'winner': 'tie', 'symbol': '≈'}

    try:
        stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False,
                'winner': 'tie', 'symbol': '≈'}

    significant = p_value < alpha
    if significant:
        winner = 'A' if np.mean(a) > np.mean(b) else 'B'
        symbol = '+' if winner == 'A' else '-'
    else:
        winner = 'tie'
        symbol = '≈'

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': significant,
        'winner': winner,
        'symbol': symbol,
    }


def pairwise_wilcoxon(results: dict, baseline_key: str,
                      metric: str = 'HV', higher_better: bool = True) -> dict:
    """对所有方法进行配对Wilcoxon检验

    Args:
        results: {method_name: [run1_value, run2_value, ...]}
        baseline_key: 基准方法名（本文方法）
        metric: 指标名称（用于日志）
        higher_better: 该指标是否越大越好

    Returns:
        {method: {'p_value', 'symbol', ...}}
    """
    baseline_data = results.get(baseline_key, [])
    comparisons = {}

    for method, data in results.items():
        if method == baseline_key:
            comparisons[method] = {'symbol': '—', 'p_value': None}
            continue

        if higher_better:
            test = wilcoxon_test(baseline_data, data)
        else:
            test = wilcoxon_test(data, baseline_data)

        comparisons[method] = test

    return comparisons


def format_mean_std(values: list, precision: int = 2) -> str:
    """格式化为 mean±std 字符串"""
    if not values:
        return "—"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def compute_win_tie_loss(comparisons: dict) -> dict:
    """统计胜/平/负次数

    Args:
        comparisons: pairwise_wilcoxon的返回结果

    Returns:
        {'win': n, 'tie': n, 'loss': n}
    """
    w, t, l = 0, 0, 0
    for method, result in comparisons.items():
        if result.get('symbol') == '+':
            w += 1
        elif result.get('symbol') == '-':
            l += 1
        elif result.get('symbol') == '≈':
            t += 1
    return {'win': w, 'tie': t, 'loss': l}
