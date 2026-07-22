"""变异算子库（5种）

对应论文第5章 5.4.2节
"""

import numpy as np
from src.algorithm.nsga3.encoding import Chromosome


def swap_mutation(chromosome: Chromosome,
                  rng: np.random.RandomState = None) -> Chromosome:
    """算子M1：工序交换变异

    随机选择OS中两个属于不同工件的位置，交换工件编号。
    """
    if rng is None:
        rng = np.random.RandomState()

    child = chromosome.copy()
    n = len(child.os)

    for _ in range(10):  # 最多尝试10次找到不同工件
        idx1, idx2 = rng.choice(n, 2, replace=False)
        if child.os[idx1] != child.os[idx2]:
            child.os[idx1], child.os[idx2] = child.os[idx2], child.os[idx1]
            break

    child.invalidate()
    return child


def insert_mutation(chromosome: Chromosome,
                    rng: np.random.RandomState = None) -> Chromosome:
    """算子M2：工序插入变异

    随机选择一个位置的元素，移除后插入到另一个随机位置。
    """
    if rng is None:
        rng = np.random.RandomState()

    child = chromosome.copy()
    n = len(child.os)

    idx_from = rng.randint(0, n)
    idx_to = rng.randint(0, n)

    if idx_from != idx_to:
        val = child.os[idx_from]
        child.os = np.delete(child.os, idx_from)
        if idx_to > idx_from:
            idx_to -= 1
        child.os = np.insert(child.os, idx_to, val)

    child.invalidate()
    return child


def machine_mutation(chromosome: Chromosome, mutation_rate: float = 0.1,
                     rng: np.random.RandomState = None) -> Chromosome:
    """算子M3：机器重分配变异

    随机选择部分工序，重新从可选机器中随机选择。
    """
    if rng is None:
        rng = np.random.RandomState()

    child = chromosome.copy()
    inst = child.instance

    idx = 0
    for i in range(inst.num_jobs):
        for j in range(inst.num_operations[i]):
            if rng.random() < mutation_rate:
                machines = inst.get_compatible_machines(i, j)
                child.ma[idx] = rng.randint(0, len(machines))
            idx += 1

    child.invalidate()
    return child


def agv_mutation(chromosome: Chromosome, mutation_rate: float = 0.1,
                 rng: np.random.RandomState = None) -> Chromosome:
    """算子M4：AGV重分配变异

    随机选择部分运输任务，重新分配AGV。
    """
    if rng is None:
        rng = np.random.RandomState()

    child = chromosome.copy()
    n = child.n_total

    mask = rng.random(n) < mutation_rate
    for idx in range(n):
        if mask[idx]:
            child.agv_assign[idx] = rng.randint(0, child.instance.num_agv)

    child.invalidate()
    return child


def speed_mutation(chromosome: Chromosome, mutation_rate: float = 0.1,
                   rng: np.random.RandomState = None) -> Chromosome:
    """算子M5：速度调整变异

    随机选择部分运输任务，改变速度档位。
    """
    if rng is None:
        rng = np.random.RandomState()

    child = chromosome.copy()
    n = child.n_total

    mask = rng.random(n) < mutation_rate
    for idx in range(n):
        if mask[idx]:
            child.agv_speed[idx] = rng.randint(0, child.instance.num_speeds)

    child.invalidate()
    return child


# 算子注册表
MUTATION_OPERATORS = {
    5: ('Swap', swap_mutation),
    6: ('Insert', insert_mutation),
    7: ('MachineReassign', machine_mutation),
    8: ('AGVReassign', agv_mutation),
    9: ('SpeedAdjust', speed_mutation),
}
