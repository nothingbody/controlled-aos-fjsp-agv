"""实验三：消融实验

特性：全局并行 + OMP_NUM_THREADS=1 + 增量保存 + 断点续跑 + 保存F数组
对应论文第7章 7.4节
"""

import sys, os, time

# 限制BLAS线程
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import load_benchmark_set
from src.algorithm.grl_ea import GRLEA
from src.baselines.pure_drl import PureDRL
from src.algorithm.nsga3.selection import compute_hypervolume, non_dominated_sort
from experiments.exp_utils import (ExperimentLogger, IncrementalSaver, Timer,
                                   estimate_remaining, save_pareto_front)
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

POP_SIZE = 100
MAX_GEN = 100
NUM_RUNS = 10
NUM_AGV = 3
MAX_WORKERS = 40
RESULT_FILE = 'results/tables/exp3_ablation.csv'

ABLATION_VARIANTS = {
    'GRL_EA_full': 'Full GRL-EA',
    'V1_no_GNN': 'w/o GNN (random emb)',
    'V2_no_ModuleA': 'w/o Module A',
    'V3_no_ModuleB': 'w/o Module B',
    'V4_no_EA': 'w/o EA (Pure DRL)',
    'V6_no_Speed': 'w/o Speed Selection',
    'V7_GNN_to_MLP': 'GNN -> MLP',
    'V8_no_GRL': 'w/o GRL (Pure NSGA-III)',
}


def run_variant(instance, variant_name, seed):
    start = time.time()
    if variant_name == 'GRL_EA_full':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, seed=seed, verbose=False)
    elif variant_name == 'V1_no_GNN':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, seed=seed, verbose=False)
        solver.hat_encoder = None
    elif variant_name == 'V2_no_ModuleA':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=False, seed=seed, verbose=False)
    elif variant_name == 'V3_no_ModuleB':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, top_k_improve=0, seed=seed, verbose=False)
    elif variant_name == 'V4_no_EA':
        solver_drl = PureDRL(instance, num_episodes=POP_SIZE * MAX_GEN // 10,
                             seed=seed, verbose=False)
        archive = solver_drl.run()
        return archive, time.time() - start
    elif variant_name == 'V6_no_Speed':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, seed=seed, verbose=False)
    elif variant_name == 'V7_GNN_to_MLP':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, seed=seed, verbose=False)
        try:
            from src.algorithm.grl.mlp_encoder import MLPEncoder
            solver._init_hat()
            solver.hat_encoder = MLPEncoder(hidden_dim=solver.hidden_dim).to(solver.device)
            solver.hat_encoder.eval()
        except Exception:
            solver.hat_encoder = None
    elif variant_name == 'V8_no_GRL':
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=False, seed=seed, verbose=False)
    else:
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=False, seed=seed, verbose=False)

    archive = solver.run()
    return archive, time.time() - start


def _ablation_worker(args):
    """全局并行worker"""
    inst_name, instance, var_name, seed = args
    try:
        archive, elapsed = run_variant(instance, var_name, seed)
        if archive:
            objs = np.array([c.objectives.to_array() for c in archive])
            fronts = non_dominated_sort(objs)
            nd = objs[fronts[0]]
            ref = nd.max(axis=0) * 1.2

            # 保存Pareto前沿
            save_pareto_front(nd, var_name, inst_name, seed, 'exp3_ablation')

            return (inst_name, var_name, seed, {
                'HV': compute_hypervolume(nd, ref),
                'Cmax': float(nd[:, 0].min()),
                'TEC': float(nd[:, 1].min()),
                'Time': elapsed
            })
        return (inst_name, var_name, seed, {
            'HV': 0, 'Cmax': float('inf'), 'TEC': float('inf'), 'Time': elapsed
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] {var_name} {inst_name} seed={seed}: {e}")
        traceback.print_exc()
        return (inst_name, var_name, seed, {
            'HV': 0, 'Cmax': float('inf'), 'TEC': float('inf'), 'Time': 0
        })


def main():
    log = ExperimentLogger('exp3_ablation')
    saver = IncrementalSaver(RESULT_FILE)
    timer = Timer()

    log.section('Experiment 3: Ablation Study (Global Parallel)')
    log.info(f'POP={POP_SIZE}, GEN={MAX_GEN}, RUNS={NUM_RUNS}, WORKERS={MAX_WORKERS}')
    log.info(f'OMP_NUM_THREADS=1')
    log.info(f'Existing records: {saver.count}')

    instances = load_benchmark_set('data/benchmarks/brandimarte', num_agv=NUM_AGV)
    test_instances = [(n, i) for n, i in instances if n in ['Mk01.fjs', 'Mk06.fjs', 'Mk10.fjs']]
    if not test_instances:
        test_instances = instances[:3]

    # 收集所有需要跑的任务
    all_combos = []
    for inst_name, instance in test_instances:
        for var_name, var_desc in ABLATION_VARIANTS.items():
            if saver.has_result(instance=inst_name, variant=var_name):
                continue
            all_combos.append((inst_name, instance, var_name, var_desc))

    if not all_combos:
        log.info('All experiments already completed!')
        return

    all_tasks = []
    for inst_name, instance, var_name, var_desc in all_combos:
        for seed in range(42, 42 + NUM_RUNS):
            all_tasks.append((inst_name, instance, var_name, seed))

    total_combos = len(all_combos)
    total_tasks = len(all_tasks)
    log.info(f'Remaining: {total_combos} combos, {total_tasks} tasks')
    log.info(f'Launching {min(MAX_WORKERS, total_tasks)} workers...\n')

    results_by_combo = defaultdict(list)
    completed_tasks = 0
    var_desc_map = dict(ABLATION_VARIANTS)

    n_workers = min(MAX_WORKERS, total_tasks, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_ablation_worker, t): t for t in all_tasks}

        for future in as_completed(futures):
            inst_name, var_name, seed, metrics = future.result()
            combo_key = (inst_name, var_name)
            results_by_combo[combo_key].append(metrics)
            completed_tasks += 1

            if len(results_by_combo[combo_key]) == NUM_RUNS:
                run_metrics = results_by_combo[combo_key]
                run_hvs = [r['HV'] for r in run_metrics]
                run_cmax = [r['Cmax'] for r in run_metrics]
                run_tec = [r['TEC'] for r in run_metrics]
                run_times = [r['Time'] for r in run_metrics]

                record = {
                    'instance': inst_name, 'variant': var_name,
                    'description': var_desc_map[var_name],
                    'HV_mean': np.mean(run_hvs), 'HV_std': np.std(run_hvs),
                    'Cmax_mean': np.mean(run_cmax), 'TEC_mean': np.mean(run_tec),
                    'Time_mean': np.mean(run_times),
                }
                saver.append(record)

                log.info(f'  [{saver.count}] {inst_name} {var_desc_map[var_name]:30s}: '
                         f'HV={record["HV_mean"]:.1f} | Cmax={record["Cmax_mean"]:.1f} | '
                         f'Time={record["Time_mean"]:.1f}s | tasks={completed_tasks}/{total_tasks}')

    # 消融分析
    log.section('Ablation Analysis (HV change vs Full)')
    df = saver.get_dataframe()
    for inst_name in df['instance'].unique():
        inst_df = df[df['instance'] == inst_name]
        full_rows = inst_df[inst_df['variant'] == 'GRL_EA_full']
        if len(full_rows) == 0:
            continue
        full_hv = full_rows['HV_mean'].values[0]
        log.info(f'\n  {inst_name} (Full HV={full_hv:.1f}):')
        for _, row in inst_df.iterrows():
            if row['variant'] != 'GRL_EA_full':
                change = (row['HV_mean'] - full_hv) / max(full_hv, 1) * 100
                log.info(f'    {row["description"]:30s}: {change:+.1f}%')

    log.info(f'\nTotal time: {timer.elapsed()}')


if __name__ == '__main__':
    main()
