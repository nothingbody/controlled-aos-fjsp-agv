"""解码器：将三层编码转化为完整调度方案

对应论文第5章 Algorithm 1: Decode
"""

import numpy as np
from dataclasses import dataclass, field
from src.problem.instance import FJSPAGVInstance
from src.problem.energy_model import compute_agv_energy, compute_machine_energy


@dataclass
class Schedule:
    """调度方案"""
    # 每道工序的时间信息 {(i,j): (start, end)}
    op_start: dict = field(default_factory=dict)
    op_end: dict = field(default_factory=dict)
    # 每道工序分配的机器
    op_machine: dict = field(default_factory=dict)
    # 运输信息 {(i,j): (transport_start, transport_end, agv_id, speed)}
    transport_start: dict = field(default_factory=dict)
    transport_end: dict = field(default_factory=dict)
    transport_agv: dict = field(default_factory=dict)
    transport_speed: dict = field(default_factory=dict)
    transport_empty_distance: dict = field(default_factory=dict)
    transport_loaded_distance: dict = field(default_factory=dict)
    # 每台机器的加工序列 {machine: [(job, op, start, end), ...]}
    machine_schedule: dict = field(default_factory=dict)
    # 每台AGV的任务序列
    agv_schedule: dict = field(default_factory=dict)


@dataclass
class Objectives:
    """目标函数值"""
    makespan: float = 0.0          # f1: 最大完工时间
    total_energy: float = 0.0      # f2: 总能耗
    workload_balance: float = 0.0  # f3: 机器负载不均衡度
    stability: float = 0.0         # f4: 重调度不稳定度（仅动态）

    def to_array(self) -> np.ndarray:
        return np.array([self.makespan, self.total_energy, self.workload_balance])

    def to_array_with_stability(self) -> np.ndarray:
        return np.array([self.makespan, self.total_energy,
                         self.workload_balance, self.stability])


def decode(chromosome) -> Schedule:
    """解码染色体为调度方案

    对应论文 Algorithm 1

    Args:
        chromosome: Chromosome对象

    Returns:
        Schedule对象
    """
    inst = chromosome.instance
    os_seq = chromosome.os
    ma_seq = chromosome.ma
    agv_assign_seq = chromosome.agv_assign
    agv_speed_seq = chromosome.agv_speed

    schedule = Schedule()
    for u in range(inst.num_machines):
        schedule.machine_schedule[u] = []
    for l in range(inst.num_agv):
        schedule.agv_schedule[l] = []

    # 资源可用时间
    machine_avail = np.zeros(inst.num_machines)
    agv_avail = np.zeros(inst.num_agv)
    agv_loc = np.zeros(inst.num_agv, dtype=int)  # 0=装载站

    # 每台机器上最后加工的工件编号（用于判断换模）
    machine_last_job = [-1] * inst.num_machines

    # 工序计数器（用字典支持动态新工件）
    op_count = {}

    # 建立 (i, j) -> idx 的映射
    op_to_idx = {}
    idx = 0
    for i in range(inst.num_jobs):
        for j in range(inst.num_operations[i]):
            op_to_idx[(i, j)] = idx
            idx += 1
    # 对于OS中可能出现的新工件，动态建立映射
    new_op_count = {}
    for t in range(len(os_seq)):
        job_id = os_seq[t]
        new_op_count[job_id] = new_op_count.get(job_id, 0) + 1
    extra_idx = idx
    for job_id in sorted(set(os_seq)):
        if job_id >= inst.num_jobs:
            for j in range(new_op_count.get(job_id, 0)):
                op_to_idx[(job_id, j)] = extra_idx
                extra_idx += 1

    # 按OS顺序解码
    for t in range(len(os_seq)):
        i = os_seq[t]  # 工件编号
        j = op_count.get(i, 0)  # 当前工序序号
        op_count[i] = j + 1

        # 查找分配的机器、AGV、速度
        flat_idx = op_to_idx.get((i, j))
        if flat_idx is None or flat_idx >= len(ma_seq):
            # 新工件的工序可能超出原始编码长度，跳过
            continue
        machines = inst.get_compatible_machines(i, j)
        if not machines:
            # 新工件可能不在instance的compatible_machines中，使用所有机器
            machines = list(range(inst.num_machines))
        ma_idx = ma_seq[flat_idx]
        u = machines[ma_idx % len(machines)]  # 机器编号
        l = agv_assign_seq[flat_idx] % inst.num_agv  # AGV编号
        speed_idx = agv_speed_seq[flat_idx] % inst.num_speeds
        speed = inst.speed_levels[speed_idx]

        # 确定运输起点
        if j == 0:
            u_prev = 0  # 装载站
        else:
            u_prev = schedule.op_machine[(i, j - 1)]
            # 距离矩阵中机器索引需要+1（0为装载站）
            u_prev = u_prev + 1

        u_dest = u + 1  # 距离矩阵索引

        # 计算AGV空载行驶时间
        dist_empty = inst.get_distance(agv_loc[l], u_prev)
        t_empty = dist_empty / speed if speed > 0 else 0

        agv_ready = agv_avail[l] + t_empty

        # 运输开始时间
        if j == 0:
            cargo_ready = 0.0
        else:
            cargo_ready = schedule.op_end[(i, j - 1)]

        transport_start = max(cargo_ready, agv_ready)

        # 运输时间
        dist_transport = inst.get_distance(u_prev, u_dest)
        t_transport = dist_transport / speed if speed > 0 else 0
        transport_end = transport_start + t_transport

        # 换模时间
        setup_time = 0.0
        if machine_last_job[u] >= 0 and machine_last_job[u] != i:
            if inst.machine_setup_time is not None:
                setup_time = inst.machine_setup_time[u]

        # 加工开始和完成时间
        proc_time = inst.get_processing_time(i, j, u)
        if proc_time == float('inf'):
            proc_time = 10.0  # 新工件默认加工时间
        start_time = max(transport_end, machine_avail[u] + setup_time)
        end_time = start_time + proc_time

        # 记录调度信息
        schedule.op_start[(i, j)] = start_time
        schedule.op_end[(i, j)] = end_time
        schedule.op_machine[(i, j)] = u
        schedule.transport_start[(i, j)] = transport_start
        schedule.transport_end[(i, j)] = transport_end
        schedule.transport_agv[(i, j)] = l
        schedule.transport_speed[(i, j)] = speed
        schedule.transport_empty_distance[(i, j)] = dist_empty
        schedule.transport_loaded_distance[(i, j)] = dist_transport

        schedule.machine_schedule[u].append((i, j, start_time, end_time))
        schedule.agv_schedule[l].append((i, j, transport_start, transport_end))

        # 更新资源状态
        machine_avail[u] = end_time
        agv_avail[l] = transport_end
        agv_loc[l] = u_dest
        machine_last_job[u] = i

    return schedule


def evaluate(chromosome, ref_schedule: Schedule = None) -> Objectives:
    """评估染色体的目标函数值

    Args:
        chromosome: Chromosome对象
        ref_schedule: 参考调度（用于计算稳定性目标f4）

    Returns:
        Objectives对象
    """
    inst = chromosome.instance
    schedule = chromosome.schedule
    obj = Objectives()

    # f1: makespan — 从schedule中所有工序结束时间取最大值
    max_end = 0.0
    if schedule.op_end:
        max_end = max(schedule.op_end.values())
    obj.makespan = max_end

    # f2: normalized total energy index
    machine_workloads = np.zeros(inst.num_machines)
    for u in range(inst.num_machines):
        machine_workloads[u] = sum(
            end - start
            for _, _, start, end in schedule.machine_schedule[u]
        )
    machine_energy = compute_machine_energy(inst, schedule, obj.makespan)
    agv_energy = compute_agv_energy(inst, schedule)
    obj.total_energy = float(
        sum(item["total"] for item in machine_energy.values())
        + sum(item["total"] for item in agv_energy.values())
    )

    # f3: 负载不均衡度
    if len(machine_workloads) > 0:
        obj.workload_balance = float(machine_workloads.max() - machine_workloads.min())
    else:
        obj.workload_balance = 0.0

    # f4: 稳定性（如果有参考调度）
    if ref_schedule is not None:
        total_deviation = 0.0
        count = 0
        for key in schedule.op_start:
            if key in ref_schedule.op_start:
                total_deviation += abs(schedule.op_start[key] - ref_schedule.op_start[key])
                count += 1
        obj.stability = total_deviation / max(count, 1)

    return obj
