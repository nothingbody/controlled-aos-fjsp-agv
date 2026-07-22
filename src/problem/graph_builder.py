"""异构图构建器：将FJSP-AGV调度状态构建为三元异构图

对应论文第4章 4.2节
节点：工序(7维) + 机器(6维) + AGV(8维)
边：工艺顺序 + 可加工 + 机器顺序 + 运输连接 + AGV位置
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.decoding import Schedule


class HeteroGraphData:
    """异构图数据容器（兼容PyG的HeteroData格式）"""

    def __init__(self):
        # 节点特征
        self.x_op: Optional[torch.Tensor] = None    # (num_ops, 7)
        self.x_mac: Optional[torch.Tensor] = None   # (num_machines, 6)
        self.x_agv: Optional[torch.Tensor] = None   # (num_agv, 8)

        # 边索引 {edge_type: (2, num_edges)}
        self.edge_index: Dict[str, torch.Tensor] = {}
        # 边特征 {edge_type: (num_edges, feat_dim)}
        self.edge_attr: Dict[str, Optional[torch.Tensor]] = {}

        # 元信息
        self.num_ops = 0
        self.num_machines = 0
        self.num_agv = 0

    def to(self, device):
        """转移到指定设备"""
        self.x_op = self.x_op.to(device) if self.x_op is not None else None
        self.x_mac = self.x_mac.to(device) if self.x_mac is not None else None
        self.x_agv = self.x_agv.to(device) if self.x_agv is not None else None
        for key in self.edge_index:
            self.edge_index[key] = self.edge_index[key].to(device)
            if self.edge_attr.get(key) is not None:
                self.edge_attr[key] = self.edge_attr[key].to(device)
        return self


def build_hetero_graph(
    instance: FJSPAGVInstance,
    schedule: Optional[Schedule] = None,
    current_time: float = 0.0,
) -> HeteroGraphData:
    """构建FJSP-AGV的异构图

    Args:
        instance: 问题实例
        schedule: 调度方案（None表示初始状态）
        current_time: 当前时刻（用于动态场景）

    Returns:
        HeteroGraphData对象
    """
    graph = HeteroGraphData()
    graph.num_ops = instance.total_operations
    graph.num_machines = instance.num_machines
    graph.num_agv = instance.num_agv

    # 建立(i,j) -> 全局工序索引的映射
    op_to_idx = {}
    idx = 0
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            op_to_idx[(i, j)] = idx
            idx += 1

    # 估计Cmax用于归一化
    cmax_est = _estimate_cmax(instance)
    p_max = _get_max_processing_time(instance)
    p_sum_max = _get_max_downstream_time(instance)

    # ========== 1. 节点特征 ==========
    graph.x_op = _build_op_features(instance, schedule, op_to_idx,
                                     cmax_est, p_max, p_sum_max, current_time)
    graph.x_mac = _build_mac_features(instance, schedule, cmax_est, current_time)
    graph.x_agv = _build_agv_features(instance, schedule, cmax_est, current_time)

    # ========== 2. 边 ==========
    # 2.1 工艺顺序边 E_prec (op -> op)
    prec_src, prec_dst = [], []
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i] - 1):
            prec_src.append(op_to_idx[(i, j)])
            prec_dst.append(op_to_idx[(i, j + 1)])

    if prec_src:
        graph.edge_index['prec'] = torch.tensor([prec_src, prec_dst], dtype=torch.long)
    else:
        graph.edge_index['prec'] = torch.zeros(2, 0, dtype=torch.long)
    graph.edge_attr['prec'] = None

    # 2.2 可加工边 E_proc (op <-> machine) — 双向
    proc_op, proc_mac, proc_feat = [], [], []
    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            op_idx = op_to_idx[(i, j)]
            for u in instance.get_compatible_machines(i, j):
                proc_op.append(op_idx)
                proc_mac.append(u)
                pt = instance.get_processing_time(i, j, u)
                proc_feat.append([pt / max(p_max, 1e-6)])

    if proc_op:
        # op -> mac
        graph.edge_index['proc_o2m'] = torch.tensor([proc_op, proc_mac], dtype=torch.long)
        graph.edge_attr['proc_o2m'] = torch.tensor(proc_feat, dtype=torch.float32)
        # mac -> op (反向)
        graph.edge_index['proc_m2o'] = torch.tensor([proc_mac, proc_op], dtype=torch.long)
        graph.edge_attr['proc_m2o'] = torch.tensor(proc_feat, dtype=torch.float32)
    else:
        graph.edge_index['proc_o2m'] = torch.zeros(2, 0, dtype=torch.long)
        graph.edge_index['proc_m2o'] = torch.zeros(2, 0, dtype=torch.long)
        graph.edge_attr['proc_o2m'] = None
        graph.edge_attr['proc_m2o'] = None

    # 2.3 机器顺序边 E_seq (op -> op, 同机器)
    seq_src, seq_dst, seq_feat = [], [], []
    if schedule is not None:
        for u in range(instance.num_machines):
            ops_on_machine = schedule.machine_schedule.get(u, [])
            for t in range(len(ops_on_machine) - 1):
                i1, j1, _, e1 = ops_on_machine[t]
                i2, j2, s2, _ = ops_on_machine[t + 1]
                src_idx = op_to_idx.get((i1, j1))
                dst_idx = op_to_idx.get((i2, j2))
                if src_idx is not None and dst_idx is not None:
                    seq_src.append(src_idx)
                    seq_dst.append(dst_idx)
                    gap = max(0, s2 - e1)
                    seq_feat.append([gap / max(cmax_est, 1e-6)])

    if seq_src:
        graph.edge_index['seq'] = torch.tensor([seq_src, seq_dst], dtype=torch.long)
        graph.edge_attr['seq'] = torch.tensor(seq_feat, dtype=torch.float32)
    else:
        graph.edge_index['seq'] = torch.zeros(2, 0, dtype=torch.long)
        graph.edge_attr['seq'] = None

    # 2.4 运输连接边 E_trans (mac <-> mac)
    trans_src, trans_dst, trans_feat = [], [], []
    d_max = 1.0
    if instance.distance_matrix is not None:
        d_max = max(instance.distance_matrix.max(), 1e-6)
        v_max = max(instance.speed_levels) if instance.speed_levels else 1.0
        for u1 in range(instance.num_machines):
            for u2 in range(u1 + 1, instance.num_machines):
                d = instance.get_distance(u1 + 1, u2 + 1)  # +1因为索引0是装载站
                if d > 0:
                    trans_src.extend([u1, u2])
                    trans_dst.extend([u2, u1])
                    feat = [d / d_max, (d / v_max) / max(cmax_est, 1e-6)]
                    trans_feat.extend([feat, feat])

    if trans_src:
        graph.edge_index['trans'] = torch.tensor([trans_src, trans_dst], dtype=torch.long)
        graph.edge_attr['trans'] = torch.tensor(trans_feat, dtype=torch.float32)
    else:
        graph.edge_index['trans'] = torch.zeros(2, 0, dtype=torch.long)
        graph.edge_attr['trans'] = None

    # 2.5 AGV位置边 E_loc (agv <-> machine)
    loc_agv, loc_mac = [], []
    if schedule is not None:
        # 根据schedule推断AGV位置
        agv_last_loc = [0] * instance.num_agv  # 默认在装载站
        for l in range(instance.num_agv):
            tasks = schedule.agv_schedule.get(l, [])
            if tasks:
                last_task = tasks[-1]
                i, j = last_task[0], last_task[1]
                machine = schedule.op_machine.get((i, j), 0)
                agv_last_loc[l] = machine

        for l in range(instance.num_agv):
            u = agv_last_loc[l]
            if 0 <= u < instance.num_machines:
                loc_agv.extend([l, l])
                loc_mac.extend([u, u])
    else:
        # 初始状态：所有AGV在机器0旁
        for l in range(instance.num_agv):
            loc_agv.extend([l, l])
            loc_mac.extend([0, 0])

    if loc_agv:
        graph.edge_index['loc_a2m'] = torch.tensor([loc_agv[::2], loc_mac[::2]], dtype=torch.long)
        graph.edge_index['loc_m2a'] = torch.tensor([loc_mac[1::2], loc_agv[1::2]], dtype=torch.long)
    else:
        graph.edge_index['loc_a2m'] = torch.zeros(2, 0, dtype=torch.long)
        graph.edge_index['loc_m2a'] = torch.zeros(2, 0, dtype=torch.long)
    graph.edge_attr['loc_a2m'] = None
    graph.edge_attr['loc_m2a'] = None

    return graph


# ========== 辅助函数 ==========

def _estimate_cmax(instance: FJSPAGVInstance) -> float:
    """粗略估计Cmax（用于特征归一化）"""
    total = 0
    for i in range(instance.num_jobs):
        job_time = 0
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            if machines:
                min_pt = min(instance.get_processing_time(i, j, u) for u in machines)
                job_time += min_pt
        total = max(total, job_time)
    return max(total * 1.5, 1.0)  # 乘以系数作为保守估计


def _get_max_processing_time(instance: FJSPAGVInstance) -> float:
    if not instance.processing_times:
        return 1.0
    return max(instance.processing_times.values())


def _get_max_downstream_time(instance: FJSPAGVInstance) -> float:
    max_sum = 1.0
    for i in range(instance.num_jobs):
        s = 0
        for j in range(instance.num_operations[i]):
            machines = instance.get_compatible_machines(i, j)
            if machines:
                s += min(instance.get_processing_time(i, j, u) for u in machines)
        max_sum = max(max_sum, s)
    return max_sum


def _build_op_features(instance, schedule, op_to_idx, cmax_est, p_max, p_sum_max, current_time):
    """构建工序节点特征 (num_ops, 7)"""
    n_ops = instance.total_operations
    features = np.zeros((n_ops, 7), dtype=np.float32)

    for i in range(instance.num_jobs):
        for j in range(instance.num_operations[i]):
            idx = op_to_idx[(i, j)]
            machines = instance.get_compatible_machines(i, j)

            # 特征1: 归一化加工时间（取可选机器的最小加工时间）
            if machines:
                min_pt = min(instance.get_processing_time(i, j, u) for u in machines)
                features[idx, 0] = min_pt / max(p_max, 1e-6)

            # 特征2: 加工柔性度
            features[idx, 1] = len(machines) / max(instance.num_machines, 1)

            # 特征3: 工序相对位置
            features[idx, 2] = (j + 1) / instance.num_operations[i]

            # 特征4: 剩余工序比例
            features[idx, 3] = (instance.num_operations[i] - j) / instance.num_operations[i]

            # 特征5: 完成状态
            if schedule is not None and (i, j) in schedule.op_end:
                if schedule.op_end[(i, j)] <= current_time:
                    features[idx, 4] = 1.0  # 已完成
                elif schedule.op_start[(i, j)] <= current_time:
                    features[idx, 4] = 0.5  # 加工中
                else:
                    features[idx, 4] = 0.0  # 未开始
            else:
                features[idx, 4] = 0.0

            # 特征6: 归一化等待时间
            if schedule is not None and (i, j) in schedule.op_start:
                wait = max(0, schedule.op_start[(i, j)] - current_time)
                features[idx, 5] = wait / max(cmax_est, 1e-6)

            # 特征7: 下游最短时间和
            downstream = 0
            for jj in range(j, instance.num_operations[i]):
                m_list = instance.get_compatible_machines(i, jj)
                if m_list:
                    downstream += min(instance.get_processing_time(i, jj, u) for u in m_list)
            features[idx, 6] = downstream / max(p_sum_max, 1e-6)

    return torch.tensor(features, dtype=torch.float32)


def _build_mac_features(instance, schedule, cmax_est, current_time):
    """构建机器节点特征 (num_machines, 6)"""
    m = instance.num_machines
    features = np.zeros((m, 6), dtype=np.float32)

    wl_max = 1.0
    q_max = 1.0
    workloads = np.zeros(m)
    queues = np.zeros(m)

    if schedule is not None:
        for u in range(m):
            ops = schedule.machine_schedule.get(u, [])
            workloads[u] = sum(e - s for (_, _, s, e) in ops)
            queues[u] = sum(1 for (_, _, s, _) in ops if s > current_time)
        wl_max = max(workloads.max(), 1e-6)
        q_max = max(queues.max(), 1.0)

    p_proc_max = 1.0
    p_idle_max = 1.0
    if instance.machine_proc_power is not None:
        p_proc_max = max(instance.machine_proc_power.max(), 1e-6)
    if instance.machine_idle_power is not None:
        p_idle_max = max(instance.machine_idle_power.max(), 1e-6)

    for u in range(m):
        features[u, 0] = workloads[u] / wl_max
        # 空闲时间
        if schedule is not None:
            ops = schedule.machine_schedule.get(u, [])
            if ops:
                last_end = max(e for (_, _, _, e) in ops)
                features[u, 1] = max(0, current_time - last_end) / max(cmax_est, 1e-6)
        features[u, 2] = queues[u] / q_max
        if instance.machine_proc_power is not None:
            features[u, 3] = instance.machine_proc_power[u] / p_proc_max
        if instance.machine_idle_power is not None:
            features[u, 4] = instance.machine_idle_power[u] / p_idle_max
        # 最早可用时间
        if schedule is not None:
            ops = schedule.machine_schedule.get(u, [])
            if ops:
                avail = max(e for (_, _, _, e) in ops)
                features[u, 5] = avail / max(cmax_est, 1e-6)

    return torch.tensor(features, dtype=torch.float32)


def _build_agv_features(instance, schedule, cmax_est, current_time):
    """构建AGV节点特征 (num_agv, 8)"""
    k = instance.num_agv
    features = np.zeros((k, 8), dtype=np.float32)

    for l in range(k):
        # 默认：空闲状态 one-hot [1, 0, 0]
        features[l, 0] = 1.0  # 空闲

        if schedule is not None:
            tasks = schedule.agv_schedule.get(l, [])
            if tasks:
                last_task = tasks[-1]
                _, _, ts, te = last_task
                if ts <= current_time < te:
                    features[l, 0:3] = [0, 1, 0]  # 载货运输中
                elif te <= current_time:
                    features[l, 0:3] = [1, 0, 0]  # 空闲

                # 位置
                i, j = last_task[0], last_task[1]
                machine = schedule.op_machine.get((i, j), 0)
                features[l, 3] = machine / max(instance.num_machines, 1)

                # 任务完成时间
                features[l, 4] = te / max(cmax_est, 1e-6)

                # 累计任务数
                features[l, 7] = len(tasks)

    # 归一化任务数
    task_max = max(features[:, 7].max(), 1.0)
    features[:, 7] /= task_max

    return torch.tensor(features, dtype=torch.float32)
