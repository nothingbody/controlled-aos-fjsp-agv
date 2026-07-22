"""能耗模型

对应论文第3章 3.5节
"""

import numpy as np
from src.problem.instance import FJSPAGVInstance


def machine_active_idle_time(ops, setup_count: int, setup_time_each: float) -> float:
    """Return standby time inside a machine's active production window.

    A machine is treated as powered down before its first processing start and
    after its last completion.  Setup time is an active state and is therefore
    removed from the residual standby time.
    """
    if not ops:
        return 0.0
    first_start = min(float(start) for _, _, start, _ in ops)
    last_end = max(float(end) for _, _, _, end in ops)
    proc_time = sum(float(end) - float(start) for _, _, start, end in ops)
    active_span = max(0.0, last_end - first_start)
    return max(0.0, active_span - proc_time - setup_count * setup_time_each)


def compute_machine_energy(instance: FJSPAGVInstance, schedule, makespan: float) -> dict:
    """计算各机器的详细能耗

    Returns:
        {machine_id: {'proc': float, 'setup': float, 'idle': float, 'total': float}}
    """
    result = {}
    for u in range(instance.num_machines):
        ops = schedule.machine_schedule.get(u, [])

        # 加工能耗
        proc_time = sum(e - s for (_, _, s, e) in ops)
        proc_power = instance.machine_proc_power[u] if instance.machine_proc_power is not None else 5.0
        proc_energy = proc_power * proc_time

        # 换模次数
        setup_count = 0
        prev_job = -1
        for (i, _, _, _) in ops:
            if prev_job >= 0 and prev_job != i:
                setup_count += 1
            prev_job = i

        setup_time_each = instance.machine_setup_time[u] if instance.machine_setup_time is not None else 4.0
        setup_power = instance.machine_setup_power[u] if instance.machine_setup_power is not None else 2.5
        setup_energy = setup_power * setup_time_each * setup_count

        # Standby energy is counted only while the machine is active.
        idle_time = machine_active_idle_time(ops, setup_count, setup_time_each)
        idle_power = instance.machine_idle_power[u] if instance.machine_idle_power is not None else 1.0
        idle_energy = idle_power * idle_time

        result[u] = {
            'proc': proc_energy,
            'setup': setup_energy,
            'idle': idle_energy,
            'total': proc_energy + setup_energy + idle_energy,
            'proc_time': proc_time,
            'idle_time': idle_time,
            'active_span': (
                max(e for _, _, _, e in ops) - min(s for _, _, s, _ in ops)
                if ops else 0.0
            ),
            'setup_count': setup_count,
        }

    return result


def compute_agv_energy(instance: FJSPAGVInstance, schedule) -> dict:
    """计算各AGV的详细能耗

    Returns:
        {agv_id: {'load': float, 'empty': float, 'total': float}}
    """
    result = {l: {'load': 0.0, 'empty': 0.0, 'total': 0.0, 'distance': 0.0}
              for l in range(instance.num_agv)}

    for key in schedule.transport_start:
        i, j = key
        l = schedule.transport_agv.get(key, 0)
        default_speed = float(np.median(instance.speed_levels))
        speed = schedule.transport_speed.get(key, default_speed)

        u_from = 0 if j == 0 else schedule.op_machine[(i, j - 1)] + 1
        u_to = schedule.op_machine[(i, j)] + 1
        dist = schedule.transport_loaded_distance.get(
            key, instance.get_distance(u_from, u_to)
        )
        empty_dist = schedule.transport_empty_distance.get(key, 0.0)

        load_energy = instance.transport_energy(dist, speed, loaded=True)
        empty_energy = instance.transport_energy(empty_dist, speed, loaded=False)
        result[l]['load'] += load_energy
        result[l]['empty'] += empty_energy
        result[l]['distance'] += dist + empty_dist

    for l in range(instance.num_agv):
        result[l]['total'] = result[l]['load'] + result[l]['empty']

    return result


def speed_energy_tradeoff_analysis(instance: FJSPAGVInstance, distance: float) -> dict:
    """分析不同速度下的能耗-时间权衡

    Args:
        instance: 问题实例
        distance: 运输距离

    Returns:
        {speed: {'time': float, 'energy': float, 'power': float}}
    """
    result = {}
    for v in instance.speed_levels:
        time = distance / v
        power = instance.agv_load_power(v)
        energy = power * time
        result[v] = {'time': time, 'energy': energy, 'power': power}
    return result
