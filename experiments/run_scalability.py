"""实验四：可扩展性分析

特性：日志保存 + 增量保存 + 断点续跑
对应论文第7章 7.5节
"""

import sys, os, time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import generate_random_instance
from src.algorithm.grl_ea import GRLEA
from src.baselines.pure_nsga3 import PureNSGA3
from src.baselines.pure_drl import PureDRL
from src.algorithm.nsga3.selection import compute_hypervolume, non_dominated_sort
from experiments.exp_utils import ExperimentLogger, IncrementalSaver, Timer, estimate_remaining
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_RUNS = 10
POP_SIZE = 80
MAX_GEN = 80
RESULT_FILE = 'results/tables/exp4_scalability.csv'

SCALES = [
    {'name': 'XS', 'jobs': 10, 'machines': 5, 'agv': 2, 'ops': (2, 5)},
    {'name': 'S',  'jobs': 20, 'machines': 8, 'agv': 3, 'ops': (3, 6)},
    {'name': 'M',  'jobs': 50, 'machines': 10, 'agv': 3, 'ops': (3, 8)},
    {'name': 'L',  'jobs': 100, 'machines': 15, 'agv': 5, 'ops': (4, 8)},
]

SCALE_METHODS = {'NSGA3': 'nsga3', 'DRL': 'drl', 'GRL_EA': 'grl_ea'}


def run_scale_test(instance, method_key, seed):
    start = time.time()
    if method_key == 'nsga3':
        solver = PureNSGA3(instance, pop_size=POP_SIZE, max_gen=MAX_GEN, seed=seed, verbose=False)
        archive = solver.run()
    elif method_key == 'drl':
        solver = PureDRL(instance, num_episodes=POP_SIZE * 2, seed=seed, verbose=False)
        archive = solver.run()
    else:
        solver = GRLEA(instance, pop_size=POP_SIZE, max_gen=MAX_GEN,
                       use_grl=True, seed=seed, verbose=False)
        archive = solver.run()

    elapsed = time.time() - start
    if archive:
        objs = np.array([c.objectives.to_array() for c in archive])
        fronts = non_dominated_sort(objs)
        nd = objs[fronts[0]]
        hv = compute_hypervolume(nd, nd.max(axis=0) * 1.2)
        best_cmax = nd[:, 0].min()
    else:
        hv, best_cmax = 0, float('inf')
    return {'HV': hv, 'Cmax': best_cmax, 'Time': elapsed}


def _scale_worker(args):
    """并行 worker: (instance, method_key, seed) -> metrics dict"""
    instance, method_key, seed = args
    return run_scale_test(instance, method_key, seed)


def run_parallel_scale(instance, method_key, num_runs=NUM_RUNS):
    tasks = [(instance, method_key, 42 + r) for r in range(num_runs)]
    results = []
    with ProcessPoolExecutor(max_workers=min(num_runs, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_scale_worker, t): t for t in tasks}
        for f in as_completed(futures):
            results.append(f.result())
    return results


def main():
    log = ExperimentLogger('exp4_scalability')
    saver = IncrementalSaver(RESULT_FILE)
    timer = Timer()

    log.section('Experiment 4: Scalability Analysis')
    log.info(f'POP={POP_SIZE}, GEN={MAX_GEN}, RUNS={NUM_RUNS} (parallel)')
    log.info(f'Existing records: {saver.count}')

    total = len(SCALES) * len(SCALE_METHODS)
    done = 0

    for scale in SCALES:
        instance = generate_random_instance(scale['jobs'], scale['machines'],
                                             scale['agv'], ops_range=scale['ops'], seed=42)
        log.info(f'\n--- Scale {scale["name"]}: {scale["jobs"]}x{scale["machines"]}x{scale["agv"]} '
                 f'({instance.total_operations} ops) ---')

        for method_label, method_key in SCALE_METHODS.items():
            done += 1
            if saver.has_result(scale=scale['name'], method=method_label):
                log.info(f'  {method_label:10s}: SKIPPED')
                continue

            run_results = run_parallel_scale(instance, method_key, NUM_RUNS)

            record = {
                'scale': scale['name'], 'jobs': scale['jobs'],
                'machines': scale['machines'], 'agv': scale['agv'],
                'total_ops': instance.total_operations, 'method': method_label,
                'HV_mean': np.mean([r['HV'] for r in run_results]),
                'HV_std': np.std([r['HV'] for r in run_results]),
                'Cmax_mean': np.mean([r['Cmax'] for r in run_results]),
                'Time_mean': np.mean([r['Time'] for r in run_results]),
                'Time_std': np.std([r['Time'] for r in run_results]),
            }
            saver.append(record)

            eta = estimate_remaining(done, total, time.time() - timer.start_time)
            log.info(f'  {method_label:10s}: HV={record["HV_mean"]:.1f} | '
                     f'Cmax={record["Cmax_mean"]:.1f} | '
                     f'Time={record["Time_mean"]:.1f}s | ETA={eta}')

    log.info(f'\nTotal time: {timer.elapsed()}')


if __name__ == '__main__':
    main()
