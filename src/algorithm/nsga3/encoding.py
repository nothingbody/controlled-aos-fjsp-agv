"""三层编码方案：工序序列(OS) + 机器分配(MA) + AGV分配/速度(AGV)

对应论文第5章 5.2节
"""

import numpy as np
from src.problem.instance import FJSPAGVInstance


class Chromosome:
    """FJSP-AGV 染色体（三层编码）"""

    def __init__(self, instance: FJSPAGVInstance,
                 os: np.ndarray = None,
                 ma: np.ndarray = None,
                 agv_assign: np.ndarray = None,
                 agv_speed: np.ndarray = None):
        self.instance = instance

        self.os = os  # 工序序列编码
        self.ma = ma  # 机器分配编码
        self.agv_assign = agv_assign  # AGV分配
        self.agv_speed = agv_speed  # AGV速度档位

        # 解码后的调度结果（懒计算）
        self._schedule = None
        self._objectives = None

    @property
    def n_total(self):
        return len(self.os) if self.os is not None else 0

    @property
    def schedule(self):
        if self._schedule is None:
            from src.algorithm.nsga3.decoding import decode
            self._schedule = decode(self)
        return self._schedule

    @property
    def objectives(self):
        if self._objectives is None:
            from src.algorithm.nsga3.decoding import evaluate
            self._objectives = evaluate(self)
        return self._objectives

    def invalidate(self):
        """编码被修改后，清除缓存的解码结果"""
        self._schedule = None
        self._objectives = None

    def copy(self) -> 'Chromosome':
        return Chromosome(
            instance=self.instance,
            os=self.os.copy(),
            ma=self.ma.copy(),
            agv_assign=self.agv_assign.copy(),
            agv_speed=self.agv_speed.copy(),
        )


def random_chromosome(instance: FJSPAGVInstance, rng: np.random.RandomState = None) -> Chromosome:
    """随机生成一个合法染色体"""
    if rng is None:
        rng = np.random.RandomState()

    n_total = instance.total_operations

    # OS: 基于工件编号的排列
    os = []
    for i in range(instance.num_jobs):
        os.extend([i] * instance.num_operations[i])
    os = np.array(os)
    rng.shuffle(os)

    # MA: 每道工序随机选择一台可选机器（存储索引）
    ma = np.zeros(n_total, dtype=int)
    idx = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            ma[idx] = rng.randint(0, len(machines))
            idx += 1

    # AGV: 随机分配AGV和速度
    agv_assign = rng.randint(0, instance.num_agv, size=n_total)
    agv_speed = rng.randint(0, instance.num_speeds, size=n_total)

    return Chromosome(instance, os, ma, agv_assign, agv_speed)


def spt_chromosome(instance: FJSPAGVInstance, rng: np.random.RandomState = None) -> Chromosome:
    """基于SPT规则生成染色体（最短加工时间优先）"""
    if rng is None:
        rng = np.random.RandomState()

    n_total = instance.total_operations

    # OS: 按工序最短加工时间排序
    ops_info = []
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            min_pt = min(instance.get_processing_time(i, j, u) for u in machines)
            ops_info.append((i, j, min_pt))
    ops_info.sort(key=lambda x: x[2])

    os_list = []
    op_count = [0] * instance.num_jobs
    for (i, j, _) in ops_info:
        os_list.append(i)
        op_count[i] += 1

    # 修复OS使工艺约束自动满足（基于工件编号的排列自动满足）
    os = np.array(os_list)

    # MA: 选择加工时间最短的机器
    ma = np.zeros(n_total, dtype=int)
    idx = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            times = [instance.get_processing_time(i, j, u) for u in machines]
            ma[idx] = int(np.argmin(times))
            idx += 1

    # AGV: 随机 + 中速
    agv_assign = rng.randint(0, instance.num_agv, size=n_total)
    agv_speed = np.ones(n_total, dtype=int)  # 索引1=中速

    return Chromosome(instance, os, ma, agv_assign, agv_speed)


def energy_chromosome(instance: FJSPAGVInstance, rng: np.random.RandomState = None) -> Chromosome:
    """基于能耗导向生成染色体"""
    if rng is None:
        rng = np.random.RandomState()

    n_total = instance.total_operations
    os = []
    for i in range(instance.num_jobs):
        os.extend([i] * instance.num_operations[i])
    os = np.array(os)
    rng.shuffle(os)

    # MA: 选择功率最低的机器
    # Processing-energy heuristic: minimize P_proc[u] * p_iju.
    ma = np.zeros(n_total, dtype=int)
    idx = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            if instance.machine_proc_power is not None:
                processing_energies = [
                    instance.machine_proc_power[u]
                    * instance.get_processing_time(i, j, u)
                    for u in machines
                ]
                ma[idx] = int(np.argmin(processing_energies))
            else:
                ma[idx] = rng.randint(0, len(machines))
            idx += 1

    # AGV: 随机分配 + 低速（节能）
    agv_assign = rng.randint(0, instance.num_agv, size=n_total)
    agv_speed = np.zeros(n_total, dtype=int)  # 索引0=低速

    return Chromosome(instance, os, ma, agv_assign, agv_speed)


def balance_chromosome(instance: FJSPAGVInstance, rng: np.random.RandomState = None) -> Chromosome:
    """基于负载均衡导向生成染色体"""
    if rng is None:
        rng = np.random.RandomState()

    n_total = instance.total_operations
    os = []
    for i in range(instance.num_jobs):
        os.extend([i] * instance.num_operations[i])
    os = np.array(os)
    rng.shuffle(os)

    # MA: 贪心选择当前负载最低的机器
    machine_load = np.zeros(instance.num_machines)
    ma = np.zeros(n_total, dtype=int)
    idx = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            loads = [machine_load[u] for u in machines]
            best_idx = int(np.argmin(loads))
            ma[idx] = best_idx
            best_machine = machines[best_idx]
            machine_load[best_machine] += instance.get_processing_time(i, j, best_machine)
            idx += 1

    agv_assign = rng.randint(0, instance.num_agv, size=n_total)
    agv_speed = np.ones(n_total, dtype=int)  # 中速

    return Chromosome(instance, os, ma, agv_assign, agv_speed)
