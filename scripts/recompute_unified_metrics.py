"""从保存的pkl重算统一参考点下的HV/IGD/Spread + Friedman/Nemenyi检验。

关键：每个实例上，所有方法共用一个参考点 = 合并所有方法F后的最差点 × 1.1。
这样HV跨方法可比。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from src.algorithm.nsga3.selection import compute_hypervolume, non_dominated_sort
from src.utils.metrics import compute_igd, compute_spread


def load_all_pkl(pkl_dir):
    """加载所有pkl，直接从pkl内部读取instance/method/seed"""
    data = []
    for pkl_path in Path(pkl_dir).glob('*.pkl'):
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        instance = str(d['instance']).replace('.fjs', '')
        method = d['method']
        seed = d['seed']
        data.append({
            'instance': instance,
            'method': method,
            'seed': seed,
            'objectives': d['objectives'],
        })
    return data


def compute_unified_metrics(data, n_obj=3):
    """对每个实例，用统一参考点计算HV/IGD/Spread"""
    # 按instance分组
    by_instance = defaultdict(list)
    for d in data:
        by_instance[d['instance']].append(d)

    records = []
    for inst, runs in sorted(by_instance.items()):
        # 合并该实例所有方法所有seed的Pareto前沿
        all_F = []
        for r in runs:
            F = r['objectives']
            if F is not None and len(F) > 0:
                all_F.append(F[:, :n_obj])  # 只取前n_obj个目标
        if not all_F:
            continue

        all_F_merged = np.vstack(all_F)

        # 统一参考点 = 全局最差点 × 1.1
        ref_point = all_F_merged.max(axis=0) * 1.1

        # 近似真实Pareto前沿 = 合并所有方法的非支配解
        fronts = non_dominated_sort(all_F_merged)
        true_pf = all_F_merged[fronts[0]]

        # 对每个(method, seed)计算指标
        for r in runs:
            F = r['objectives']
            if F is None or len(F) == 0:
                records.append({
                    'instance': inst, 'method': r['method'], 'seed': r['seed'],
                    'HV_unified': 0, 'IGD_unified': float('inf'),
                    'Spread_unified': 0, 'NSol': 0,
                    'Cmax_best': float('inf'), 'TEC_best': float('inf'),
                })
                continue

            F_obj = F[:, :n_obj]

            # 只取非支配解
            nd_fronts = non_dominated_sort(F_obj)
            nd_F = F_obj[nd_fronts[0]]

            hv = compute_hypervolume(nd_F, ref_point)
            igd = compute_igd(nd_F, true_pf)
            spread = compute_spread(nd_F) if len(nd_F) > 2 else 0

            records.append({
                'instance': inst, 'method': r['method'], 'seed': r['seed'],
                'HV_unified': hv, 'IGD_unified': igd,
                'Spread_unified': spread, 'NSol': len(nd_F),
                'Cmax_best': float(nd_F[:, 0].min()),
                'TEC_best': float(nd_F[:, 1].min()),
            })

    return pd.DataFrame(records)


def run_statistical_tests(df, metric='HV_unified', higher_is_better=True):
    """Friedman + pairwise Wilcoxon检验"""
    from scipy import stats

    methods = sorted(df['method'].unique())
    instances = sorted(df['instance'].unique())

    # 构建 (instance×seed) × method 矩阵
    # 对每个(instance, seed)取值
    pivot = df.pivot_table(index=['instance', 'seed'], columns='method',
                           values=metric, aggfunc='first')
    pivot = pivot.dropna(axis=0, how='any')  # 去掉缺失行

    if len(pivot) < 3 or len(pivot.columns) < 3:
        print(f"Not enough data for Friedman test (rows={len(pivot)}, methods={len(pivot.columns)})")
        return None, None

    # Friedman检验
    try:
        stat, p = stats.friedmanchisquare(*[pivot[m].values for m in pivot.columns])
        print(f"\nFriedman test on {metric}:")
        print(f"  chi2 = {stat:.2f}, p = {p:.2e}")
        print(f"  {'Significant' if p < 0.05 else 'Not significant'} (alpha=0.05)")
    except Exception as e:
        print(f"Friedman test failed: {e}")
        stat, p = None, None

    # 平均排名
    print(f"\nAverage ranks ({metric}, {'higher=better' if higher_is_better else 'lower=better'}):")
    if higher_is_better:
        ranks = pivot.rank(axis=1, ascending=False)
    else:
        ranks = pivot.rank(axis=1, ascending=True)
    avg_ranks = ranks.mean().sort_values()
    for method, rank in avg_ranks.items():
        mean_val = pivot[method].mean()
        print(f"  {method:20s}: rank={rank:.2f}, mean={mean_val:.2e}")

    # Pairwise Wilcoxon
    print(f"\nPairwise Wilcoxon signed-rank test (p-values):")
    n_methods = len(methods)
    p_matrix = pd.DataFrame(np.ones((n_methods, n_methods)),
                            index=methods, columns=methods)
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i >= j:
                continue
            try:
                s, pval = stats.wilcoxon(pivot[m1].values, pivot[m2].values)
                p_matrix.loc[m1, m2] = pval
                p_matrix.loc[m2, m1] = pval
            except Exception:
                pass

    # Win/Tie/Loss vs GRL_EA
    grl_method = [m for m in methods if 'GRL_EA' in m and 'noGRL' not in m
                  and 'full' not in m.lower()]
    if not grl_method:
        grl_method = [m for m in methods if m == 'GRL_EA_full']
    if not grl_method:
        grl_method = [m for m in methods if 'GRL_EA' in m]

    if grl_method:
        grl_m = grl_method[0]
        print(f"\nWin/Tie/Loss for {grl_m} vs each baseline ({metric}):")
        for m in methods:
            if m == grl_m:
                continue
            if higher_is_better:
                wins = (pivot[grl_m] > pivot[m]).sum()
                losses = (pivot[grl_m] < pivot[m]).sum()
            else:
                wins = (pivot[grl_m] < pivot[m]).sum()
                losses = (pivot[grl_m] > pivot[m]).sum()
            ties = len(pivot) - wins - losses
            pval = p_matrix.loc[grl_m, m]
            sig = '*' if pval < 0.05 else ''
            print(f"  vs {m:20s}: W={wins:3d} T={ties:3d} L={losses:3d}  p={pval:.4f} {sig}")

    return avg_ranks, p_matrix


def main():
    os.makedirs('results/tables', exist_ok=True)

    # ==================== Exp1 ====================
    print("=" * 70)
    print("Processing Exp1: Static Comparison")
    print("=" * 70)

    pkl_dir = 'results/pareto_fronts/exp1_static'
    data = load_all_pkl(pkl_dir)
    print(f"Loaded {len(data)} pkl files")

    df = compute_unified_metrics(data)
    df.to_csv('results/tables/exp1_unified_metrics.csv', index=False)
    print(f"Saved {len(df)} records to exp1_unified_metrics.csv")

    # 汇总表
    summary = df.groupby('method').agg({
        'HV_unified': ['mean', 'std'],
        'IGD_unified': ['mean', 'std'],
        'Spread_unified': ['mean', 'std'],
        'NSol': 'mean',
        'Cmax_best': 'mean',
        'TEC_best': 'mean',
    }).round(4)
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary = summary.sort_values('HV_unified_mean', ascending=False)
    print("\n--- Summary (sorted by unified HV, higher=better) ---")
    print(summary.to_string())
    summary.to_csv('results/tables/exp1_unified_summary.csv')

    # 统计检验
    run_statistical_tests(df, 'HV_unified', higher_is_better=True)
    print("\n--- IGD Tests ---")
    run_statistical_tests(df, 'IGD_unified', higher_is_better=False)

    # ==================== Exp3 ====================
    print("\n" + "=" * 70)
    print("Processing Exp3: Ablation Study")
    print("=" * 70)

    pkl_dir3 = 'results/pareto_fronts/exp3_ablation'
    data3 = load_all_pkl(pkl_dir3)
    print(f"Loaded {len(data3)} pkl files")

    df3 = compute_unified_metrics(data3)
    df3.to_csv('results/tables/exp3_unified_metrics.csv', index=False)
    print(f"Saved {len(df3)} records to exp3_unified_metrics.csv")

    # 消融汇总
    summary3 = df3.groupby('method').agg({
        'HV_unified': ['mean', 'std'],
        'Cmax_best': 'mean',
        'TEC_best': 'mean',
    }).round(4)
    summary3.columns = ['_'.join(c).strip('_') for c in summary3.columns]

    # 计算相对变化
    full_hv = summary3.loc['GRL_EA_full', 'HV_unified_mean'] if 'GRL_EA_full' in summary3.index else None
    full_cmax = summary3.loc['GRL_EA_full', 'Cmax_best_mean'] if 'GRL_EA_full' in summary3.index else None
    if full_hv and full_hv > 0:
        summary3['HV_change_%'] = ((summary3['HV_unified_mean'] - full_hv) / full_hv * 100).round(2)
    if full_cmax and full_cmax > 0:
        summary3['Cmax_change_%'] = ((summary3['Cmax_best_mean'] - full_cmax) / full_cmax * 100).round(2)

    summary3 = summary3.sort_values('HV_unified_mean', ascending=False)
    print("\n--- Ablation Summary (unified HV) ---")
    print(summary3.to_string())
    summary3.to_csv('results/tables/exp3_unified_summary.csv')

    print("\n\nDONE! All unified metrics saved.")


if __name__ == '__main__':
    main()
