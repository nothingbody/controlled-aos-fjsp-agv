"""实验五：参数敏感性分析

特性：日志保存 + 增量保存 + 断点续跑
对应论文第7章 7.6节
"""

import sys, os, time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import load_fjsp_file
from src.algorithm.grl_ea import GRLEA
from src.algorithm.nsga3.selection import compute_hypervolume, non_dominated_sort
from experiments.exp_utils import ExperimentLogger, IncrementalSaver, Timer, estimate_remaining
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_RUNS = 10
TEST_INSTANCE = 'data/benchmarks/brandimarte/Mk06.fjs'
NUM_AGV = 3
RESULT_FILE = 'results/tables/exp5_sensitivity.csv'

PARAM_TESTS = {
    'pop_size': {'values': [50, 100, 200, 300], 'default': 100, 'key': 'pop_size'},
    'max_gen': {'values': [50, 100, 200, 500], 'default': 100, 'key': 'max_gen'},
    'improve_steps': {'values': [0, 1, 3, 5, 8], 'default': 5, 'key': 'improve_steps'},
    'top_k_improve': {'values': [5, 10, 20, 30], 'default': 20, 'key': 'top_k_improve'},
}

DEFAULT_PARAMS = {'pop_size': 100, 'max_gen': 100, 'improve_steps': 5, 'top_k_improve': 20}


def run_with_params(instance, params, seed):
    start = time.time()
    solver = GRLEA(instance, pop_size=params['pop_size'], max_gen=params['max_gen'],
                   top_k_improve=params['top_k_improve'], improve_steps=params['improve_steps'],
                   use_grl=True, seed=seed, verbose=False)
    archive = solver.run()
    elapsed = time.time() - start

    if archive:
        objs = np.array([c.objectives.to_array() for c in archive])
        fronts = non_dominated_sort(objs)
        nd = objs[fronts[0]]
        hv = compute_hypervolume(nd, nd.max(axis=0) * 1.2)
        best_cmax = nd[:, 0].min()
        best_tec = nd[:, 1].min()
    else:
        hv, best_cmax, best_tec = 0, float('inf'), float('inf')

    return {'HV': hv, 'Cmax': best_cmax, 'TEC': best_tec, 'Time': elapsed}


def _sens_worker(args):
    """并行 worker: (instance, params_dict, seed) -> metrics dict"""
    instance, params, seed = args
    return run_with_params(instance, params, seed)


def run_parallel_sensitivity(instance, params, num_runs=NUM_RUNS):
    tasks = [(instance, params, 42 + r) for r in range(num_runs)]
    results = []
    with ProcessPoolExecutor(max_workers=min(num_runs, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_sens_worker, t): t for t in tasks}
        for f in as_completed(futures):
            results.append(f.result())
    return results


def main():
    log = ExperimentLogger('exp5_sensitivity')
    saver = IncrementalSaver(RESULT_FILE)
    timer = Timer()

    log.section('Experiment 5: Parameter Sensitivity Analysis')
    instance = load_fjsp_file(TEST_INSTANCE, num_agv=NUM_AGV)
    log.info(f'Instance: {instance.summary()}')
    log.info(f'RUNS={NUM_RUNS} (parallel)')

    total = sum(len(c['values']) for c in PARAM_TESTS.values())
    done = 0

    for param_name, config in PARAM_TESTS.items():
        log.info(f'\n--- Parameter: {param_name} ---')
        for val in config['values']:
            done += 1
            if saver.has_result(parameter=param_name, value=val):
                log.info(f'  {param_name}={val:>5}: SKIPPED')
                continue

            params = DEFAULT_PARAMS.copy()
            params[config['key']] = val

            run_results = run_parallel_sensitivity(instance, params, NUM_RUNS)

            record = {
                'parameter': param_name, 'value': val,
                'is_default': val == config['default'],
                'HV_mean': np.mean([r['HV'] for r in run_results]),
                'HV_std': np.std([r['HV'] for r in run_results]),
                'Cmax_mean': np.mean([r['Cmax'] for r in run_results]),
                'TEC_mean': np.mean([r['TEC'] for r in run_results]),
                'Time_mean': np.mean([r['Time'] for r in run_results]),
            }
            saver.append(record)

            marker = " <-- default" if record['is_default'] else ""
            eta = estimate_remaining(done, total, time.time() - timer.start_time)
            log.info(f'  {param_name}={val:>5}: HV={record["HV_mean"]:.1f} | '
                     f'Time={record["Time_mean"]:.1f}s | ETA={eta}{marker}')

    log.section('Sensitivity Summary')
    df = saver.get_dataframe()
    for pn in PARAM_TESTS:
        pdf = df[df['parameter'] == pn]
        if len(pdf) > 0:
            hv_range = pdf['HV_mean'].max() - pdf['HV_mean'].min()
            hv_mean = pdf['HV_mean'].mean()
            sens = hv_range / max(hv_mean, 1) * 100
            log.info(f'  {pn:20s}: HV variation = {sens:.1f}%')

    log.info(f'\nTotal time: {timer.elapsed()}')


if __name__ == '__main__':
    main()
