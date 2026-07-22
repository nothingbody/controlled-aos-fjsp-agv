"""一键运行全部5组实验

特性：
  - 每个实验独立运行，一个失败不影响其他
  - 支持断点续跑（每个实验内部自动跳过已完成的记录）
  - 总控日志保存

用法：
  python experiments/run_all.py          # 运行全部
  python experiments/run_all.py 1        # 只运行实验1
  python experiments/run_all.py 1 3 5    # 运行实验1、3、5
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp_utils import ExperimentLogger, Timer

EXPERIMENTS = {
    1: ('Exp1: Static Comparison', 'experiments/run_static.py'),
    2: ('Exp2: Dynamic Comparison', 'experiments/run_dynamic.py'),
    3: ('Exp3: Ablation Study', 'experiments/run_ablation.py'),
    4: ('Exp4: Scalability', 'experiments/run_scalability.py'),
    5: ('Exp5: Sensitivity', 'experiments/run_sensitivity.py'),
}


def main():
    if len(sys.argv) > 1:
        exp_ids = [int(x) for x in sys.argv[1:] if x.isdigit()]
    else:
        exp_ids = list(EXPERIMENTS.keys())

    log = ExperimentLogger('run_all')
    timer = Timer()

    log.section('GRL-EA Experiment Runner')
    log.info(f'Experiments to run: {exp_ids}')
    log.info(f'Each experiment supports checkpoint resume.')
    log.info('')

    results = {}

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            log.info(f'Experiment {exp_id} not found, skipping.')
            continue

        name, script = EXPERIMENTS[exp_id]
        log.info(f'\n{"=" * 70}')
        log.info(f'Starting [{exp_id}] {name}')
        log.info(f'Script: {script}')
        log.info(f'{"=" * 70}')

        start = time.time()
        ret = os.system(f'{sys.executable} {script}')
        elapsed = time.time() - start

        status = 'OK' if ret == 0 else f'FAILED (code={ret})'
        results[exp_id] = {'name': name, 'status': status, 'time': elapsed}

        log.info(f'\n[{exp_id}] {name} -- {status}, time={elapsed:.1f}s')

    # 总结
    log.section('Summary')
    for eid, r in results.items():
        log.info(f'  [{eid}] {r["name"]:30s} {r["status"]:10s} {r["time"]:.1f}s')

    log.info(f'\nTotal time: {timer.elapsed()}')

    # 列出结果文件
    log.info(f'\nResult files:')
    if os.path.exists('results/tables'):
        for f in sorted(os.listdir('results/tables')):
            fpath = os.path.join('results/tables', f)
            size = os.path.getsize(fpath)
            log.info(f'  {f} ({size} bytes)')

    log.info(f'\nLog files:')
    if os.path.exists('results/logs'):
        for f in sorted(os.listdir('results/logs')):
            if f.endswith('.log'):
                log.info(f'  results/logs/{f}')


if __name__ == '__main__':
    main()
