"""实验一：静态FJSP-AGV对比实验

特性：
  - 全局并行：所有(实例×方法×seed)任务一次性提交到进程池
  - OMP_NUM_THREADS=1 防止BLAS抢核
  - 运行日志自动保存到 results/logs/exp1_*.log
  - 结果增量保存，每完成一个(实例×方法)就写入CSV
  - 支持断点续跑：检测已完成的记录自动跳过
  - 保存原始Pareto前沿F数组用于后处理统一HV

对应论文第7章 7.2节
"""

import sys, os, time

# 限制BLAS线程——必须在import numpy之前设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import load_benchmark_set
from src.algorithm.grl_ea import GRLEA
from src.baselines.dispatching_rules import multi_rule_solve
from src.baselines.pure_nsga3 import PureNSGA3, PureSPEA2, PureMOEAD
from src.baselines.pure_drl import PureDRL
from src.baselines.rl_ea_baselines import HRLMA, KEARL
from src.algorithm.nsga3.selection import compute_hypervolume, non_dominated_sort
from src.utils.metrics import compute_igd, compute_spread
from experiments.exp_utils import (ExperimentLogger, IncrementalSaver, Timer,
                                   estimate_remaining, save_pareto_front)
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# ====== 实验参数 ======
POP_SIZE = 100
MAX_GEN = 100
NUM_RUNS = 10
NUM_AGV = 3
MAX_WORKERS = 40  # 全局并行worker数
RESULT_FILE = 'results/tables/exp1_static_comparison.csv'

DATASETS = {
    'Brandimarte': 'data/benchmarks/brandimarte',
    'Hurink_edata': 'data/benchmarks/hurink_edata',
}

METHODS = {
    'B1_Rules': 'dispatching_rules',
    'B2_NSGA3': 'pure_nsga3',
    'B3_SPEA2': 'pure_spea2',
    'B4_MOEAD': 'pure_moead',
    'B5_DRL': 'pure_drl',
    'B6_HRLMA': 'hrlma',
    'B7_KEARL': 'kearl',
    'GRL_EA': 'grl_ea',
    'GRL_EA_noGRL': 'grl_ea_nogrl',
}


def run_method(method_name, instance, seed):
    """运行指定方法并返回Pareto解集和运行时间"""
    start = time.time()

    if method_name == 'dispatching_rules':
        archive = multi_rule_solve(instance, seed=seed)
    elif method_name == 'pure_nsga3':
        solver = PureNSGA3(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                           seed=seed, verbose=False)
        archive = solver.run()
    elif method_name == 'pure_spea2':
        solver = PureSPEA2(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                           seed=seed, verbose=False)
        archive = solver.run()
    elif method_name == 'pure_moead':
        solver = PureMOEAD(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                           seed=seed, verbose=False)
        archive = solver.run()
    elif method_name == 'pure_drl':
        solver = PureDRL(instance, num_episodes=POP_SIZE * 2, seed=seed, verbose=False)
        archive = solver.run()
    elif method_name == 'hrlma':
        solver = HRLMA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       seed=seed, verbose=False)
        archive = solver.run()
    elif method_name == 'kearl':
        solver = KEARL(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       seed=seed, verbose=False)
        archive = solver.run()
    elif method_name == 'grl_ea':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, seed=seed, verbose=False, device='cpu')
        archive = solver.run()
    elif method_name == 'grl_ea_nogrl':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=False, seed=seed, verbose=False, device='cpu')
        archive = solver.run()
    else:
        archive = []

    elapsed = time.time() - start
    return archive, elapsed


def evaluate_archive(archive):
    """评估Pareto解集的各项指标"""
    if not archive:
        return {'HV': 0, 'Spread': 0, 'Cmax_best': float('inf'),
                'TEC_best': float('inf'), 'WB_best': float('inf'), 'n_solutions': 0}

    objs = np.array([c.objectives.to_array() for c in archive])
    fronts = non_dominated_sort(objs)
    nd_objs = objs[fronts[0]]

    ref_point = nd_objs.max(axis=0) * 1.2
    hv = compute_hypervolume(nd_objs, ref_point)
    spread = compute_spread(nd_objs) if len(nd_objs) > 2 else 0

    return {
        'HV': hv, 'Spread': spread,
        'Cmax_best': float(nd_objs[:, 0].min()),
        'TEC_best': float(nd_objs[:, 1].min()),
        'WB_best': float(nd_objs[:, 2].min()),
        'n_solutions': len(nd_objs),
    }


def _single_run_worker(args):
    """全局并行worker: 执行单次运行并返回结果"""
    ds_name, inst_name, method_label, method_key, instance, seed = args
    try:
        archive, elapsed = run_method(method_key, instance, seed)
        metrics = evaluate_archive(archive)
        metrics['time'] = elapsed

        # 保存原始Pareto前沿
        if archive:
            objs = np.array([c.objectives.to_array() for c in archive])
            fronts = non_dominated_sort(objs)
            nd_objs = objs[fronts[0]]
            save_pareto_front(nd_objs, method_label, inst_name, seed, 'exp1_static')

        return (ds_name, inst_name, method_label, seed, metrics)
    except Exception as e:
        import traceback
        print(f"[ERROR] {method_label} {inst_name} seed={seed}: {e}")
        traceback.print_exc()
        return (ds_name, inst_name, method_label, seed, {
            'HV': 0, 'Spread': 0, 'Cmax_best': float('inf'),
            'TEC_best': float('inf'), 'WB_best': float('inf'),
            'n_solutions': 0, 'time': 0,
        })


def main():
    log = ExperimentLogger('exp1_static')
    saver = IncrementalSaver(RESULT_FILE)
    timer = Timer()

    log.section('Experiment 1: Static FJSP-AGV Comparison (Global Parallel)')
    log.info(f'POP={POP_SIZE}, GEN={MAX_GEN}, RUNS={NUM_RUNS}, WORKERS={MAX_WORKERS}, AGV={NUM_AGV}')
    log.info(f'OMP_NUM_THREADS=1 (BLAS single-thread per worker)')
    log.info(f'Result file: {RESULT_FILE}')
    log.info(f'Existing records: {saver.count} (will skip)')

    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/pareto_fronts/exp1_static', exist_ok=True)

    # 收集所有需要跑的(dataset, instance, method)组合
    all_combos = []
    for ds_name, ds_path in DATASETS.items():
        instances = load_benchmark_set(ds_path, num_agv=NUM_AGV)
        for inst_name, instance in instances:
            for method_label, method_key in METHODS.items():
                if saver.has_result(dataset=ds_name, instance=inst_name, method=method_label):
                    continue
                all_combos.append((ds_name, inst_name, method_label, method_key, instance))

    if not all_combos:
        log.info('All experiments already completed! Nothing to do.')
        return

    # 展开为所有(combo × seed)的任务列表
    all_tasks = []
    for ds_name, inst_name, method_label, method_key, instance in all_combos:
        for seed in range(42, 42 + NUM_RUNS):
            all_tasks.append((ds_name, inst_name, method_label, method_key, instance, seed))

    total_combos = len(all_combos)
    total_tasks = len(all_tasks)
    log.info(f'Remaining: {total_combos} (instance×method) combos, {total_tasks} total tasks')
    log.info(f'Launching {min(MAX_WORKERS, total_tasks)} parallel workers...\n')

    # 收集结果：按(ds_name, inst_name, method_label)分组
    results_by_combo = defaultdict(list)
    completed_tasks = 0

    n_workers = min(MAX_WORKERS, total_tasks, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_single_run_worker, t): t for t in all_tasks}

        for future in as_completed(futures):
            ds_name, inst_name, method_label, seed, metrics = future.result()
            combo_key = (ds_name, inst_name, method_label)
            results_by_combo[combo_key].append(metrics)
            completed_tasks += 1

            # 当某个combo的所有seed都完成时，汇总并保存
            if len(results_by_combo[combo_key]) == NUM_RUNS:
                run_metrics = results_by_combo[combo_key]
                avg = {k: np.mean([m[k] for m in run_metrics]) for k in run_metrics[0]}
                std = {k: np.std([m[k] for m in run_metrics]) for k in run_metrics[0]}

                record = {
                    'dataset': ds_name, 'instance': inst_name, 'method': method_label,
                    'HV_mean': avg['HV'], 'HV_std': std['HV'],
                    'Spread_mean': avg['Spread'], 'Spread_std': std['Spread'],
                    'Cmax_mean': avg['Cmax_best'], 'Cmax_std': std['Cmax_best'],
                    'TEC_mean': avg['TEC_best'], 'TEC_std': std['TEC_best'],
                    'WB_mean': avg['WB_best'], 'WB_std': std['WB_best'],
                    'Time_mean': avg['time'], 'Time_std': std['time'],
                    'NSol_mean': avg['n_solutions'],
                }
                saver.append(record)

                combos_done = saver.count
                eta = estimate_remaining(combos_done, combos_done + total_combos - combos_done,
                                         time.time() - timer.start_time)
                log.info(f'  [{combos_done}] {ds_name}/{inst_name} {method_label:15s}: '
                         f'HV={avg["HV"]:.1f}±{std["HV"]:.1f} | '
                         f'Cmax={avg["Cmax_best"]:.1f} | TEC={avg["TEC_best"]:.1f} | '
                         f'Time={avg["time"]:.1f}s | tasks={completed_tasks}/{total_tasks}')

    # 最终汇总
    log.section('Results Summary')
    df = saver.get_dataframe()
    if len(df) > 0:
        ranking = df.groupby('method')['HV_mean'].mean().sort_values(ascending=False)
        for rank, (method, hv) in enumerate(ranking.items(), 1):
            log.info(f'  #{rank}: {method:15s} HV={hv:.1f}')

    log.info(f'\nTotal time: {timer.elapsed()}')
    log.info(f'Results saved to: {RESULT_FILE}')
    log.info(f'Total records: {saver.count}')


if __name__ == '__main__':
    main()
