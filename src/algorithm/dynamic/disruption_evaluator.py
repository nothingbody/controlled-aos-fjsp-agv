"""扰动程度评估模块

对应论文第6章 6.2节 式(96)-(106)
"""

import numpy as np
from typing import Set, Tuple
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.decoding import Schedule
from src.environment.dynamic_events import DynamicEvent, EventType


def find_affected_operations(
    event: DynamicEvent,
    schedule: Schedule,
    instance: FJSPAGVInstance,
    current_time: float,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """识别受影响的工序

    Args:
        event: 动态事件
        schedule: 当前调度方案
        instance: 问题实例
        current_time: 当前时刻

    Returns:
        (直接影响集, 间接影响集)
    """
    direct = set()
    indirect = set()

    if event.event_type == EventType.NEW_JOB:
        # 新工件的所有工序
        for j in range(event.new_num_ops):
            direct.add((event.new_job_id, j))

        # 间接影响：与新工件共享机器的未开始工序
        if event.new_compatible_machines:
            new_machines = set()
            for key, machines in event.new_compatible_machines.items():
                new_machines.update(machines)

            for i in range(instance.num_jobs):
                for j in range(instance.num_operations[i]):
                    if (i, j) in schedule.op_start and schedule.op_start[(i, j)] > current_time:
                        op_machine = schedule.op_machine.get((i, j))
                        if op_machine is not None and op_machine in new_machines:
                            indirect.add((i, j))

    elif event.event_type == EventType.MACHINE_BREAKDOWN:
        u = event.machine_id
        # 直接影响：在故障机器上正在加工或排队的工序
        for i in range(instance.num_jobs):
            for j in range(instance.num_operations[i]):
                if (i, j) not in schedule.op_machine:
                    continue
                if schedule.op_machine[(i, j)] != u:
                    continue
                start = schedule.op_start.get((i, j), float('inf'))
                end = schedule.op_end.get((i, j), float('inf'))

                if start <= current_time < end:
                    direct.add((i, j))  # 正在加工（被中断）
                elif start > current_time:
                    direct.add((i, j))  # 排队等待

        # 间接影响：直接受影响工序的所有后续工序
        for (i, j) in list(direct):
            if i < len(instance.num_operations):
                for jj in range(j + 1, instance.num_operations[i]):
                    indirect.add((i, jj))

    elif event.event_type == EventType.AGV_BREAKDOWN:
        l = event.agv_id
        # 直接影响：由故障AGV执行的运输任务对应的工序
        for key, agv in schedule.transport_agv.items():
            if agv != l:
                continue
            i, j = key
            ts = schedule.transport_start.get(key, float('inf'))
            te = schedule.transport_end.get(key, float('inf'))

            if ts <= current_time < te:
                direct.add((i, j))  # 正在运输
            elif ts > current_time:
                direct.add((i, j))  # 待运输

        # 间接影响：后续工序
        for (i, j) in list(direct):
            if i < len(instance.num_operations) and j + 1 < instance.num_operations[i]:
                indirect.add((i, j + 1))

    return direct, indirect


def evaluate_disruption(
    event: DynamicEvent,
    schedule: Schedule,
    instance: FJSPAGVInstance,
    current_time: float,
    weights: tuple = (0.30, 0.25, 0.30, 0.15),
    indirect_decay: float = 0.5,
) -> float:
    """量化评估扰动程度

    对应论文式(102)

    Args:
        event: 动态事件
        schedule: 当前调度方案
        instance: 问题实例
        current_time: 当前时刻
        weights: (w1, w2, w3, w4) 各分量权重
        indirect_decay: 间接影响衰减系数 beta

    Returns:
        扰动程度 D ∈ [0, 1]
    """
    w1, w2, w3, w4 = weights

    direct, indirect = find_affected_operations(event, schedule, instance, current_time)

    # 剩余未完成工序数
    n_remaining = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            if (i, j) not in schedule.op_end or schedule.op_end[(i, j)] > current_time:
                n_remaining += 1
    # 加上新工件的工序
    if event.event_type == EventType.NEW_JOB:
        n_remaining += event.new_num_ops

    n_remaining = max(n_remaining, 1)

    # R_ops: 受影响工序比例
    r_ops = (len(direct) + indirect_decay * len(indirect)) / n_remaining

    # R_mac: 受影响机器比例
    affected_machines = set()
    for (i, j) in direct:
        if (i, j) in schedule.op_machine:
            affected_machines.add(schedule.op_machine[(i, j)])
    if event.event_type == EventType.MACHINE_BREAKDOWN:
        affected_machines.add(event.machine_id)
    r_mac = len(affected_machines) / max(instance.num_machines, 1)

    # U_time: 工期紧迫度
    cmax_plan = 0
    for i in range(instance.num_jobs):
        last = instance.num_operations[i] - 1
        if (i, last) in schedule.op_end:
            cmax_plan = max(cmax_plan, schedule.op_end[(i, last)])

    remaining_time = max(cmax_plan - current_time, 1.0)

    if event.event_type == EventType.NEW_JOB:
        # 新工件最短加工路径时间
        if event.new_compatible_machines and event.new_processing_times:
            min_path = 0
            for j in range(event.new_num_ops):
                machines = event.new_compatible_machines.get((event.new_job_id, j), [])
                if machines:
                    min_pt = min(
                        event.new_processing_times.get((event.new_job_id, j, u), float('inf'))
                        for u in machines
                    )
                    min_path += min_pt
            delay_est = min_path
        else:
            delay_est = 10.0
    elif event.event_type == EventType.MACHINE_BREAKDOWN:
        delay_est = event.repair_time
    else:  # AGV故障
        delay_est = event.agv_repair_time + 5.0  # 加上运输延迟估计

    u_time = delay_est / remaining_time

    # R_agv: AGV损失比例
    r_agv = (1.0 / max(instance.num_agv, 1)) if event.event_type == EventType.AGV_BREAKDOWN else 0.0

    # 综合扰动程度
    D = w1 * r_ops + w2 * r_mac + w3 * u_time + w4 * r_agv
    D = np.clip(D, 0.0, 1.0)

    return float(D)
