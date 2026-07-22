"""交叉算子库（5种）

对应论文第5章 5.4.1节
"""

import numpy as np
from src.algorithm.nsga3.encoding import Chromosome


def pox_crossover(p1: Chromosome, p2: Chromosome,
                  rng: np.random.RandomState = None) -> tuple:
    """算子C1：POX交叉（Precedence Operation Crossover）

    随机选择工件子集，保留父代1中该子集工序的位置，
    用父代2的其余工序填充。
    """
    if rng is None:
        rng = np.random.RandomState()

    inst = p1.instance
    # 如果长度不同则退化为复制
    if len(p1.os) != len(p2.os):
        return p1.copy(), p2.copy()

    all_jobs = sorted(set(p1.os) & set(p2.os))
    n_jobs = len(all_jobs)
    if n_jobs < 2:
        return p1.copy(), p2.copy()

    # 随机选择工件子集
    subset_size = rng.randint(1, n_jobs)
    job_subset = set(rng.choice(all_jobs, subset_size, replace=False))

    # 子代1
    c1_os = _pox_build(p1.os, p2.os, job_subset)
    c2_os = _pox_build(p2.os, p1.os, job_subset)

    # MA和AGV继承对应父代
    c1 = Chromosome(inst, c1_os, p1.ma.copy(), p1.agv_assign.copy(), p1.agv_speed.copy())
    c2 = Chromosome(inst, c2_os, p2.ma.copy(), p2.agv_assign.copy(), p2.agv_speed.copy())

    return c1, c2


def _pox_build(os_keep, os_fill, job_subset):
    """POX构建辅助函数"""
    child = np.full_like(os_keep, -1)
    # 保留job_subset中工件的位置
    for idx in range(len(os_keep)):
        if os_keep[idx] in job_subset:
            child[idx] = os_keep[idx]
    # 从os_fill中按顺序填充其余位置
    fill_ops = [os_fill[idx] for idx in range(len(os_fill)) if os_fill[idx] not in job_subset]
    fill_idx = 0
    for idx in range(len(child)):
        if child[idx] == -1:
            if fill_idx < len(fill_ops):
                child[idx] = fill_ops[fill_idx]
                fill_idx += 1
            else:
                # 防御性：用os_keep中的值填充
                child[idx] = os_keep[idx]
    return child


def jbx_crossover(p1: Chromosome, p2: Chromosome,
                  rng: np.random.RandomState = None) -> tuple:
    """算子C2：JBX交叉（Job-Based Crossover）

    将工件分为两组，分别从两个父代继承完整信息。
    """
    if rng is None:
        rng = np.random.RandomState()

    inst = p1.instance
    # 如果长度不同则退化为复制
    if len(p1.os) != len(p2.os):
        return p1.copy(), p2.copy()

    # 从OS中动态获取工件集合（支持新工件）
    all_jobs = sorted(set(p1.os) & set(p2.os))  # 取交集确保两者都有
    n_jobs = len(all_jobs)
    if n_jobs < 2:
        return p1.copy(), p2.copy()

    # 随机分组
    perm = rng.permutation(n_jobs)
    split = rng.randint(1, n_jobs)
    group_a = set(all_jobs[i] for i in perm[:split])

    c1_os, c1_ma, c1_agv_a, c1_agv_s = _jbx_build(p1, p2, group_a, inst)
    c2_os, c2_ma, c2_agv_a, c2_agv_s = _jbx_build(p2, p1, group_a, inst)

    c1 = Chromosome(inst, c1_os, c1_ma, c1_agv_a, c1_agv_s)
    c2 = Chromosome(inst, c2_os, c2_ma, c2_agv_a, c2_agv_s)
    return c1, c2


def _jbx_build(p_main, p_other, group_a, inst):
    """JBX构建辅助函数"""
    os_main_a = [p_main.os[i] for i in range(len(p_main.os)) if p_main.os[i] in group_a]
    os_other_b = [p_other.os[i] for i in range(len(p_other.os)) if p_other.os[i] not in group_a]

    child_os = []
    idx_a, idx_b = 0, 0
    for t in range(len(p_main.os)):
        if p_main.os[t] in group_a:
            if idx_a < len(os_main_a):
                child_os.append(os_main_a[idx_a])
                idx_a += 1
            else:
                child_os.append(p_main.os[t])
        else:
            if idx_b < len(os_other_b):
                child_os.append(os_other_b[idx_b])
                idx_b += 1
            else:
                child_os.append(p_main.os[t])

    child_os = np.array(child_os)
    # MA和AGV：group_a的工序从p_main继承，其余从p_other继承
    child_ma = p_main.ma.copy()
    child_agv_a = p_main.agv_assign.copy()
    child_agv_s = p_main.agv_speed.copy()

    idx = 0
    for i in range(inst.num_jobs):
        for j in range(inst.num_operations[i]):
            if i not in group_a:
                child_ma[idx] = p_other.ma[idx]
                child_agv_a[idx] = p_other.agv_assign[idx]
                child_agv_s[idx] = p_other.agv_speed[idx]
            idx += 1

    return child_os, child_ma, child_agv_a, child_agv_s


def uniform_machine_crossover(p1: Chromosome, p2: Chromosome,
                              rng: np.random.RandomState = None) -> tuple:
    """算子C3：均匀机器交叉

    OS不变，MA以概率0.5从两个父代继承。
    """
    if rng is None:
        rng = np.random.RandomState()

    inst = p1.instance
    if len(p1.ma) != len(p2.ma):
        return p1.copy(), p2.copy()
    mask = rng.random(p1.n_total) < 0.5

    c1_ma = np.where(mask, p1.ma, p2.ma)
    c2_ma = np.where(mask, p2.ma, p1.ma)

    c1 = Chromosome(inst, p1.os.copy(), c1_ma, p1.agv_assign.copy(), p1.agv_speed.copy())
    c2 = Chromosome(inst, p2.os.copy(), c2_ma, p2.agv_assign.copy(), p2.agv_speed.copy())
    return c1, c2


def uniform_agv_crossover(p1: Chromosome, p2: Chromosome,
                          rng: np.random.RandomState = None) -> tuple:
    """算子C4：均匀AGV交叉

    OS和MA不变，AGV分配和速度以概率0.5继承。
    """
    if rng is None:
        rng = np.random.RandomState()

    if len(p1.agv_assign) != len(p2.agv_assign):
        return p1.copy(), p2.copy()
    mask = rng.random(p1.n_total) < 0.5

    c1_agv_a = np.where(mask, p1.agv_assign, p2.agv_assign)
    c1_agv_s = np.where(mask, p1.agv_speed, p2.agv_speed)
    c2_agv_a = np.where(mask, p2.agv_assign, p1.agv_assign)
    c2_agv_s = np.where(mask, p2.agv_speed, p1.agv_speed)

    inst = p1.instance
    c1 = Chromosome(inst, p1.os.copy(), p1.ma.copy(), c1_agv_a, c1_agv_s)
    c2 = Chromosome(inst, p2.os.copy(), p2.ma.copy(), c2_agv_a, c2_agv_s)
    return c1, c2


def two_point_crossover(p1: Chromosome, p2: Chromosome,
                        rng: np.random.RandomState = None) -> tuple:
    """算子C5：两点交叉

    在OS上选两个点交叉，并修复可行性。
    """
    if rng is None:
        rng = np.random.RandomState()

    inst = p1.instance
    # 如果长度不同则退化为复制+变异
    if len(p1.os) != len(p2.os):
        return p1.copy(), p2.copy()
    n = len(p1.os)

    pt1, pt2 = sorted(rng.choice(n, 2, replace=False))

    c1_os = _two_point_build(p1.os, p2.os, pt1, pt2, inst)
    c2_os = _two_point_build(p2.os, p1.os, pt1, pt2, inst)

    c1 = Chromosome(inst, c1_os, p1.ma.copy(), p1.agv_assign.copy(), p1.agv_speed.copy())
    c2 = Chromosome(inst, c2_os, p2.ma.copy(), p2.agv_assign.copy(), p2.agv_speed.copy())
    return c1, c2


def _two_point_build(os1, os2, pt1, pt2, inst):
    """两点交叉构建+修复"""
    child = os1.copy()
    child[pt1:pt2] = os2[pt1:pt2]

    # 修复：确保每个工件出现正确次数
    target_count = {}
    for i in range(inst.num_jobs):
        target_count[i] = inst.num_operations[i]

    current_count = {}
    for v in child:
        current_count[v] = current_count.get(v, 0) + 1

    excess = []
    deficit = []
    for i in range(inst.num_jobs):
        diff = current_count.get(i, 0) - target_count[i]
        if diff > 0:
            excess.extend([i] * diff)
        elif diff < 0:
            deficit.extend([i] * (-diff))

    if excess and deficit:
        np.random.shuffle(deficit)
        d_idx = 0
        for idx in range(len(child)):
            if child[idx] in excess:
                # 检查是否还需要替换
                cur = child[idx]
                cur_count = sum(1 for x in child[:idx+1] if x == cur)
                if cur_count > target_count[cur]:
                    child[idx] = deficit[d_idx]
                    d_idx += 1
                    if d_idx >= len(deficit):
                        break

    return child


# 算子注册表
CROSSOVER_OPERATORS = {
    0: ('POX', pox_crossover),
    1: ('JBX', jbx_crossover),
    2: ('UniformMA', uniform_machine_crossover),
    3: ('UniformAGV', uniform_agv_crossover),
    4: ('TwoPoint', two_point_crossover),
}
