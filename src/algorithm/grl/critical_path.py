"""关键路径识别与瓶颈分析

对应论文第5章 5.6.1节 Algorithm 2: FindCriticalPath
"""

import numpy as np
from src.algorithm.nsga3.decoding import Schedule
from src.problem.instance import FJSPAGVInstance


def find_critical_path(schedule: Schedule, instance: FJSPAGVInstance) -> list:
    """识别调度方案的关键路径

    构建调度DAG，用拓扑排序+DP找最长路径。

    Args:
        schedule: 调度方案
        instance: 问题实例

    Returns:
        关键路径上的工序列表 [(job, op), ...]
    """
    if not schedule.op_start:
        return []

    # 收集所有工序节点
    all_ops = list(schedule.op_start.keys())
    if not all_ops:
        return []

    # 为每个节点计算加工时间+运输时间（边权）
    node_weight = {}
    for (i, j) in all_ops:
        proc_time = schedule.op_end[(i, j)] - schedule.op_start[(i, j)]
        trans_time = 0.0
        if (i, j) in schedule.transport_end and (i, j) in schedule.transport_start:
            trans_time = schedule.transport_end[(i, j)] - schedule.transport_start[(i, j)]
        node_weight[(i, j)] = proc_time + trans_time

    # 构建邻接表（有向边）
    adj = {op: [] for op in all_ops}
    in_degree = {op: 0 for op in all_ops}

    # 工艺约束边
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i] - 1):
            if (i, j) in schedule.op_start and (i, j + 1) in schedule.op_start:
                adj[(i, j)].append((i, j + 1))
                in_degree[(i, j + 1)] = in_degree.get((i, j + 1), 0) + 1

    # 机器约束边（同机器上相邻工序）
    for u in range(instance.num_machines):
        ops_on_m = schedule.machine_schedule.get(u, [])
        for t in range(len(ops_on_m) - 1):
            op1 = (ops_on_m[t][0], ops_on_m[t][1])
            op2 = (ops_on_m[t + 1][0], ops_on_m[t + 1][1])
            if op1 in adj and op2 in in_degree:
                adj[op1].append(op2)
                in_degree[op2] = in_degree.get(op2, 0) + 1

    # 拓扑排序 + 最长路径DP
    from collections import deque
    queue = deque()
    dist = {op: 0.0 for op in all_ops}
    prev = {op: None for op in all_ops}

    for op in all_ops:
        if in_degree.get(op, 0) == 0:
            queue.append(op)
            dist[op] = node_weight.get(op, 0)

    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj.get(u, []):
            new_dist = dist[u] + node_weight.get(v, 0)
            if new_dist > dist[v]:
                dist[v] = new_dist
                prev[v] = u
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # 找最长路径终点
    if not dist:
        return []
    end_node = max(dist, key=dist.get)

    # 回溯关键路径
    path = []
    node = end_node
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path


def find_bottleneck_machine(schedule: Schedule, instance: FJSPAGVInstance) -> int:
    """找到负载最重的瓶颈机器"""
    workloads = np.zeros(instance.num_machines)
    for u in range(instance.num_machines):
        for (_, _, s, e) in schedule.machine_schedule.get(u, []):
            workloads[u] += e - s
    return int(np.argmax(workloads))


def compute_bottleneck_features(schedule: Schedule, instance: FJSPAGVInstance,
                                 cmax: float) -> np.ndarray:
    """计算瓶颈特征向量

    对应论文式(87): [关键路径占比, 瓶颈机器负载, 最大AGV等待, 能耗集中度]
    """
    total_ops = sum(instance.num_operations[i] for i in range(instance.num_jobs))

    # 关键路径占比
    critical = find_critical_path(schedule, instance)
    cp_ratio = len(critical) / max(total_ops, 1)

    # 瓶颈机器负载
    workloads = np.zeros(instance.num_machines)
    for u in range(instance.num_machines):
        for (_, _, s, e) in schedule.machine_schedule.get(u, []):
            workloads[u] += e - s
    wl_max = workloads.max() if len(workloads) > 0 else 0
    bottleneck_load = wl_max / max(cmax, 1e-6)

    # 最大AGV等待时间
    max_agv_wait = 0.0
    for l in range(instance.num_agv):
        tasks = schedule.agv_schedule.get(l, [])
        for t in range(1, len(tasks)):
            gap = tasks[t][2] - tasks[t - 1][3]  # start_next - end_prev
            max_agv_wait = max(max_agv_wait, gap)
    agv_wait_norm = max_agv_wait / max(cmax, 1e-6)

    # 能耗集中度（基尼系数）
    if workloads.sum() > 0:
        sorted_wl = np.sort(workloads)
        n = len(sorted_wl)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_wl) - (n + 1) * np.sum(sorted_wl)) / (n * np.sum(sorted_wl))
    else:
        gini = 0.0

    return np.array([cp_ratio, bottleneck_load, agv_wait_norm, gini], dtype=np.float32)
