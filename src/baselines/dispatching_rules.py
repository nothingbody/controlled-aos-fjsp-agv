"""调度规则基线方法 (B1)

实现SPT、FIFO、MOPNR、EDD等经典调度规则
"""

import numpy as np
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.encoding import Chromosome
from src.algorithm.nsga3.decoding import decode, evaluate, Objectives


def dispatching_rule_solve(instance: FJSPAGVInstance, rule: str = 'SPT',
                           machine_rule: str = 'SPT', seed: int = 42) -> Chromosome:
    """用调度规则生成调度方案

    Args:
        instance: 问题实例
        rule: 工序选择规则 ('SPT', 'FIFO', 'MOPNR', 'LPT', 'RANDOM')
        machine_rule: 机器选择规则 ('SPT', 'MIN_LOAD', 'MIN_POWER', 'RANDOM')
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)
    n_total = instance.total_operations

    # 生成工序优先级排序 → OS编码
    ops_priority = []
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            min_pt = min(instance.get_processing_time(i, j, u) for u in machines) if machines else 999

            if rule == 'SPT':
                priority = min_pt
            elif rule == 'LPT':
                priority = -min_pt
            elif rule == 'MOPNR':
                priority = -(instance.num_operations[i] - j)  # 剩余工序多的优先
            elif rule == 'FIFO':
                priority = i * 100 + j  # 按工件和工序自然顺序
            elif rule == 'RANDOM':
                priority = rng.random()
            else:
                priority = min_pt

            ops_priority.append((i, j, priority))

    ops_priority.sort(key=lambda x: x[2])

    # 构建OS：按优先级排列工件编号
    os_seq = np.array([op[0] for op in ops_priority])

    # MA编码：按machine_rule选择机器
    ma_seq = np.zeros(n_total, dtype=int)
    machine_load = np.zeros(instance.num_machines)
    idx = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            if not machines:
                ma_seq[idx] = 0
                idx += 1
                continue

            if machine_rule == 'SPT':
                times = [instance.get_processing_time(i, j, u) for u in machines]
                ma_seq[idx] = int(np.argmin(times))
            elif machine_rule == 'MIN_LOAD':
                loads = [machine_load[u] for u in machines]
                best = int(np.argmin(loads))
                ma_seq[idx] = best
                machine_load[machines[best]] += instance.get_processing_time(i, j, machines[best])
            elif machine_rule == 'MIN_POWER':
                if instance.machine_proc_power is not None:
                    powers = [instance.machine_proc_power[u] for u in machines]
                    ma_seq[idx] = int(np.argmin(powers))
                else:
                    ma_seq[idx] = rng.randint(0, len(machines))
            else:
                ma_seq[idx] = rng.randint(0, len(machines))
            idx += 1

    # AGV: 轮询分配 + 中速
    agv_assign = np.array([t % instance.num_agv for t in range(n_total)])
    agv_speed = np.ones(n_total, dtype=int)  # 中速

    chrom = Chromosome(instance, os_seq, ma_seq, agv_assign, agv_speed)
    _ = chrom.objectives
    return chrom


def multi_rule_solve(instance: FJSPAGVInstance, seed: int = 42) -> list:
    """用多组规则组合生成一组非支配解"""
    rules = [
        ('SPT', 'SPT'), ('SPT', 'MIN_LOAD'), ('SPT', 'MIN_POWER'),
        ('LPT', 'SPT'), ('LPT', 'MIN_LOAD'),
        ('MOPNR', 'SPT'), ('MOPNR', 'MIN_LOAD'),
        ('FIFO', 'SPT'), ('FIFO', 'MIN_LOAD'),
        ('RANDOM', 'RANDOM'),
    ]
    results = []
    for op_rule, mac_rule in rules:
        for s in range(3):  # 每组规则3个随机种子
            chrom = dispatching_rule_solve(instance, op_rule, mac_rule, seed + s)
            results.append(chrom)
    return results
