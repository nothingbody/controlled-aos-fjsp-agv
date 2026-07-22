"""实验二：动态调度对比实验

特性：日志保存 + 增量保存 + 断点续跑
对应论文第7章 7.3节
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
from src.environment.dynamic_events import EventGenerator
from src.algorithm.dynamic.rescheduler import GradedRescheduler
from src.baselines.right_shift import RightShiftRescheduler, FullRescheduler
from experiments.exp_utils import ExperimentLogger, IncrementalSaver, Timer, estimate_remaining

# ====== 实验参数 ======
NUM_RUNS = 10
INIT_POP = 50
INIT_GEN = 50
MAX_EVENTS = 5
RESULT_FILE = 'results/tables/exp2_dynamic_comparison.csv'

SCENARIOS = {
    'A': {'new_job_rate': 0.2, 'machine_rate': 0.0, 'agv_rate': 0.0, 'name': 'NewJob Only'},
    'B': {'new_job_rate': 0.0, 'machine_rate': 0.03, 'agv_rate': 0.0, 'name': 'MachineBreak Only'},
    'C': {'new_job_rate': 0.0, 'machine_rate': 0.0, 'agv_rate': 0.03, 'name': 'AGVBreak Only'},
    'D': {'new_job_rate': 0.1, 'machine_rate': 0.01, 'agv_rate': 0.01, 'name': 'Mixed'},
}

DYNAMIC_METHODS = {
    'RightShift': 'right_shift',
    'FullReschedule': 'full_reschedule',
    'GradedResponse': 'graded',
}


def generate_initial_schedule(instance, seed):
    solver = GRLEA(instance, pop_size=INIT_POP, max_gen=INIT_GEN,
                   use_grl=False, seed=seed, verbose=False)
    archive = solver.run()
    return min(archive, key=lambda c: c.objectives.makespan)


def run_dynamic_scenario(instance, scenario_params, method_key, init_chrom, seed):
    start = time.time()
    gen = EventGenerator(instance, new_job_rate=scenario_params['new_job_rate'],
                         machine_breakdown_rate=scenario_params['machine_rate'],
                         agv_breakdown_rate=scenario_params['agv_rate'], seed=seed)

    cmax_init = init_chrom.objectives.makespan
    events = gen.generate_events(time_horizon=cmax_init, start_time=cmax_init * 0.2)
    events = events[:MAX_EVENTS]

    if not events:
        return {'num_events': 0, 'cmax_final': cmax_init, 'cmax_deviation': 0.0,
                'tec_final': init_chrom.objectives.total_energy,
                'avg_response_time': 0.0, 'time': time.time() - start}

    if method_key == 'right_shift':
        rescheduler = RightShiftRescheduler(instance, seed=seed)
    elif method_key == 'full_reschedule':
        rescheduler = FullRescheduler(instance, pop_size=50, max_gen=50, seed=seed)
    else:
        rescheduler = GradedRescheduler(instance, theta1=0.15, theta2=0.40,
                                         local_pop_size=30, local_max_gen=50,
                                         global_pop_size=50, global_max_gen=80, seed=seed)

    current = init_chrom
    response_times = []
    for event in events:
        t0 = time.time()
        current = rescheduler.respond(event=event, current_schedule=current.schedule,
                                      current_chromosome=current, current_time=event.time)
        response_times.append(time.time() - t0)

    elapsed = time.time() - start
    cmax_final = current.objectives.makespan
    return {
        'num_events': len(events), 'cmax_final': cmax_final,
        'cmax_deviation': (cmax_final - cmax_init) / cmax_init * 100,
        'tec_final': current.objectives.total_energy,
        'avg_response_time': np.mean(response_times),
        'time': elapsed,
    }


def main():
    log = ExperimentLogger('exp2_dynamic')
    saver = IncrementalSaver(RESULT_FILE)
    timer = Timer()

    log.section('Experiment 2: Dynamic Scheduling Comparison')
    log.info(f'INIT_POP={INIT_POP}, INIT_GEN={INIT_GEN}, MAX_EVENTS={MAX_EVENTS}, RUNS={NUM_RUNS}')
    log.info(f'Existing records: {saver.count}')

    instance = generate_random_instance(15, 8, 3, ops_range=(3, 6), pt_range=(1, 15), seed=42)
    log.info(f'Instance: {instance.summary()}')

    total = len(SCENARIOS) * len(DYNAMIC_METHODS) * NUM_RUNS
    done = 0

    for scenario_id, sp in SCENARIOS.items():
        log.info(f'\n--- Scenario {scenario_id}: {sp["name"]} ---')
        for method_label, method_key in DYNAMIC_METHODS.items():
            if saver.has_result(scenario=scenario_id, method=method_label):
                log.info(f'  {method_label:20s}: SKIPPED')
                done += NUM_RUNS
                continue

            run_results = []
            for run in range(NUM_RUNS):
                done += 1
                seed = 42 + run
                try:
                    init_chrom = generate_initial_schedule(instance, seed)
                    result = run_dynamic_scenario(instance, sp, method_key, init_chrom, seed)
                    run_results.append(result)
                except Exception as e:
                    log.info(f'  {method_label} run{run} ERROR: {e}')
                    run_results.append({'cmax_deviation': 0, 'avg_response_time': 0, 'time': 0})

            record = {
                'scenario': scenario_id, 'scenario_name': sp['name'], 'method': method_label,
                'Cmax_deviation_%_mean': np.mean([r['cmax_deviation'] for r in run_results]),
                'avg_response_time_mean': np.mean([r['avg_response_time'] for r in run_results]),
                'total_time_mean': np.mean([r['time'] for r in run_results]),
            }
            saver.append(record)

            eta = estimate_remaining(done, total, time.time() - timer.start_time)
            log.info(f'  {method_label:20s}: Cmax_dev={record["Cmax_deviation_%_mean"]:+.1f}% | '
                     f'resp={record["avg_response_time_mean"]:.3f}s | ETA={eta}')

    log.info(f'\nTotal time: {timer.elapsed()}')
    log.info(f'Results: {RESULT_FILE} ({saver.count} records)')

    # 自动可视化
    try:
        import os; os.makedirs('results/figures', exist_ok=True)
        from src.utils.result_analysis import plot_boxplot
        df = saver.get_dataframe()
        if len(df) > 0:
            dev_by_method = {}
            for method in df['method'].unique():
                dev_by_method[method] = df[df['method'] == method]['Cmax_deviation_%_mean'].tolist()
            plot_boxplot(dev_by_method, ylabel='Cmax Deviation (%)',
                         title='Dynamic Comparison: Cmax Deviation',
                         save_path='results/figures/exp2_cmax_deviation.png')
            log.info('Figure: results/figures/exp2_cmax_deviation.png')
    except Exception as e:
        log.info(f'Visualization failed: {e}')


if __name__ == '__main__':
    main()
