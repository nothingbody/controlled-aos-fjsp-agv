"""分级响应重调度器

对应论文第6章 6.3-6.6节 Algorithm 5-9
"""

import numpy as np
from typing import Optional

from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.encoding import Chromosome, random_chromosome
from src.algorithm.nsga3.decoding import Schedule, decode, evaluate
from src.algorithm.nsga3.crossover import CROSSOVER_OPERATORS
from src.algorithm.nsga3.mutation import MUTATION_OPERATORS
from src.algorithm.nsga3.selection import non_dominated_sort, nsga3_select, generate_reference_points
from src.algorithm.dynamic.disruption_evaluator import (
    evaluate_disruption, find_affected_operations
)
from src.environment.dynamic_events import DynamicEvent, EventType


class GradedRescheduler:
    """分级响应重调度器

    三级响应：
    - Level 1 (D < theta1): RL快速响应
    - Level 2 (theta1 <= D < theta2): 局部EA重调度
    - Level 3 (D >= theta2): 全局GRL-EA重调度
    """

    def __init__(self, instance: FJSPAGVInstance,
                 theta1: float = 0.15,
                 theta2: float = 0.40,
                 local_pop_size: int = 50,
                 local_max_gen: int = 100,
                 global_pop_size: int = 100,
                 global_max_gen: int = 200,
                 seed: int = 42):
        self.instance = instance
        self.theta1 = theta1
        self.theta2 = theta2
        self.local_pop_size = local_pop_size
        self.local_max_gen = local_max_gen
        self.global_pop_size = global_pop_size
        self.global_max_gen = global_max_gen
        self.rng = np.random.RandomState(seed)

        # 响应记录
        self.response_log = []

    def respond(self, event: DynamicEvent, current_schedule: Schedule,
                current_chromosome: Chromosome,
                current_time: float) -> Chromosome:
        """分级响应主入口

        对应论文 Algorithm 9

        Args:
            event: 动态事件
            current_schedule: 当前调度方案
            current_chromosome: 当前染色体编码
            current_time: 当前时刻

        Returns:
            调整后的染色体
        """
        # Step 1: 评估扰动程度
        D = evaluate_disruption(event, current_schedule, self.instance, current_time)

        # Step 2: 分级响应
        if D < self.theta1:
            level = 1
            result = self._level1_quick_response(event, current_chromosome, current_time)
        elif D < self.theta2:
            level = 2
            result = self._level2_local_reschedule(event, current_chromosome, current_time)
        else:
            level = 3
            result = self._level3_global_reschedule(event, current_chromosome, current_time)

        # 记录
        self.response_log.append({
            'time': current_time,
            'event_type': event.event_type.value,
            'disruption': D,
            'level': level,
            'makespan_before': current_chromosome.objectives.makespan,
            'makespan_after': result.objectives.makespan,
        })

        return result

    def _level1_quick_response(self, event: DynamicEvent,
                                chromosome: Chromosome,
                                current_time: float) -> Chromosome:
        """Level 1: RL快速响应

        对应论文 Algorithm 5
        Level 1 处理轻微扰动（D < theta1），策略：最小化调整幅度
        """
        child = chromosome.copy()
        inst = self.instance

        if event.event_type == EventType.NEW_JOB:
            # 将新工序插入负载最低的机器
            self._insert_new_job_simple(child, event)

        elif event.event_type == EventType.MACHINE_BREAKDOWN:
            # 轻微故障：仅在必要时做最小调整，优先等待修复
            repair_time = getattr(event, 'repair_time', 10.0)
            remaining_time = child.objectives.makespan - current_time
            if repair_time < remaining_time * 0.1:
                # 修复时间很短（<剩余工期10%），直接等待，不调整
                pass
            else:
                self._shift_affected_ops(child, event, current_time)

        elif event.event_type == EventType.AGV_BREAKDOWN:
            # 将受影响的运输任务转给最近的空闲AGV
            self._reassign_agv_tasks(child, event)

        child.invalidate()
        return child

    def _level2_local_reschedule(self, event: DynamicEvent,
                                  chromosome: Chromosome,
                                  current_time: float) -> Chromosome:
        """Level 2: 局部EA重调度

        对应论文 Algorithm 7
        对受影响的子问题进行小规模EA搜索
        """
        # 识别受影响工序
        schedule = chromosome.schedule
        direct, indirect = find_affected_operations(
            event, schedule, self.instance, current_time
        )
        affected = direct | indirect

        if not affected:
            return chromosome

        # 生成局部种群（在受影响的工序上做随机变异）
        population = []
        for _ in range(self.local_pop_size):
            child = chromosome.copy()
            # 对受影响的工序做随机扰动
            self._perturb_affected(child, affected)
            child.invalidate()
            population.append(child)

        # 快速EA搜索
        ref_points = generate_reference_points(3, 4)
        for gen in range(self.local_max_gen):
            # 随机选择算子
            op_id = self.rng.randint(0, 10)
            offspring = []

            for _ in range(self.local_pop_size // 2):
                idx1, idx2 = self.rng.choice(len(population), 2, replace=False)
                p1, p2 = population[idx1], population[idx2]

                if op_id < 5:
                    _, op_func = CROSSOVER_OPERATORS[op_id]
                    c1, c2 = op_func(p1, p2, self.rng)
                else:
                    _, op_func = MUTATION_OPERATORS[op_id]
                    if op_id >= 7:
                        c1 = op_func(p1, mutation_rate=0.15, rng=self.rng)
                        c2 = op_func(p2, mutation_rate=0.15, rng=self.rng)
                    else:
                        c1 = op_func(p1, rng=self.rng)
                        c2 = op_func(p2, rng=self.rng)
                offspring.extend([c1, c2])

            for c in offspring:
                _ = c.objectives

            combined = population + offspring
            objs = np.array([c.objectives.to_array() for c in combined])
            selected = nsga3_select(objs, self.local_pop_size, ref_points)
            population = [combined[i] for i in selected]

        # 选择最佳折中解
        best = self._select_compromise(population)
        return best

    def _level3_global_reschedule(self, event: DynamicEvent,
                                   chromosome: Chromosome,
                                   current_time: float) -> Chromosome:
        """Level 3: 全局GRL-EA重调度

        对应论文 Algorithm 8
        用完整EA（缩减参数）重新求解
        """
        # 简化版：基于当前染色体做大幅变异生成种群
        population = []
        for _ in range(self.global_pop_size):
            child = chromosome.copy()
            # 随机打乱OS
            self.rng.shuffle(child.os)
            # 随机重分配MA和AGV
            for idx in range(child.n_total):
                i_op, j_op = self._flat_to_op(self.instance, idx)
                machines = self.instance.get_compatible_machines(i_op, j_op)
                if machines:
                    child.ma[idx] = self.rng.randint(0, len(machines))
                else:
                    child.ma[idx] = self.rng.randint(0, max(self.instance.num_machines, 1))
                child.agv_assign[idx] = self.rng.randint(0, self.instance.num_agv)
                child.agv_speed[idx] = self.rng.randint(0, self.instance.num_speeds)
            child.invalidate()
            _ = child.objectives
            population.append(child)

        ref_points = generate_reference_points(3, 6)

        for gen in range(self.global_max_gen):
            op_id = self.rng.randint(0, 10)
            offspring = []

            for _ in range(self.global_pop_size // 2):
                idx1, idx2 = self.rng.choice(len(population), 2, replace=False)
                p1, p2 = population[idx1], population[idx2]

                if op_id < 5:
                    _, op_func = CROSSOVER_OPERATORS[op_id]
                    c1, c2 = op_func(p1, p2, self.rng)
                else:
                    _, op_func = MUTATION_OPERATORS[op_id]
                    if op_id >= 7:
                        c1 = op_func(p1, mutation_rate=0.1, rng=self.rng)
                        c2 = op_func(p2, mutation_rate=0.1, rng=self.rng)
                    else:
                        c1 = op_func(p1, rng=self.rng)
                        c2 = op_func(p2, rng=self.rng)
                offspring.extend([c1, c2])

            for c in offspring:
                _ = c.objectives

            combined = population + offspring
            objs = np.array([c.objectives.to_array() for c in combined])
            keep_size = min(self.global_pop_size, len(combined))
            selected = nsga3_select(objs, keep_size, ref_points)
            population = [combined[i] for i in selected]

        best = self._select_compromise(population)
        return best

    # ========== 辅助方法 ==========

    def _insert_new_job_simple(self, chromosome: Chromosome, event: DynamicEvent):
        """简单插入新工件（Level 1辅助）"""
        # 将新工件编号追加到OS末尾（简化处理）
        new_entries = np.full(event.new_num_ops, event.new_job_id)
        chromosome.os = np.concatenate([chromosome.os, new_entries])

        # MA: 选加工时间最短的机器
        new_ma = []
        for j in range(event.new_num_ops):
            machines = event.new_compatible_machines.get((event.new_job_id, j), [0])
            times = [event.new_processing_times.get((event.new_job_id, j, u), 999)
                     for u in machines]
            new_ma.append(int(np.argmin(times)))
        chromosome.ma = np.concatenate([chromosome.ma, new_ma])

        # AGV: 随机
        new_agv = self.rng.randint(0, self.instance.num_agv, size=event.new_num_ops)
        new_speed = np.ones(event.new_num_ops, dtype=int)  # 中速
        chromosome.agv_assign = np.concatenate([chromosome.agv_assign, new_agv])
        chromosome.agv_speed = np.concatenate([chromosome.agv_speed, new_speed])
        # n_total is now a property derived from len(os), no need to update

    def _shift_affected_ops(self, chromosome: Chromosome, event: DynamicEvent,
                            current_time: float):
        """右移受影响工序（Level 1辅助）

        修复：对于轻微机器故障（Level 1），不做机器重分配（那是Level 2/3的工作），
        而是仅对受影响工序做时间右移，等机器修复后继续执行。
        只有当工序在故障机器上且无法等待修复时，才重分配到其他可选机器。
        """
        u_broken = event.machine_id
        repair_time = getattr(event, 'repair_time', 10.0)
        schedule = chromosome.schedule

        idx = 0
        for i in range(self.instance.num_jobs):
            for j in range(self.instance.num_operations[i]):
                machines = self.instance.get_compatible_machines(i, j)
                current_machine_idx = chromosome.ma[idx]
                if current_machine_idx < len(machines) and machines[current_machine_idx] == u_broken:
                    # 检查是否已完成——已完成的不需要调整
                    op_end = schedule.op_end.get((i, j), float('inf'))
                    if op_end <= current_time:
                        idx += 1
                        continue

                    # Level 1策略：优先等待修复（不重分配），仅在有更优替代时才迁移
                    alternatives = [m for m_idx2, m in enumerate(machines)
                                    if m != u_broken]
                    if alternatives:
                        # 选择加工时间最短的替代机器
                        best_alt = min(alternatives,
                                       key=lambda m: self.instance.get_processing_time(i, j, m))
                        best_time = self.instance.get_processing_time(i, j, best_alt)
                        current_time_on_broken = self.instance.get_processing_time(i, j, u_broken)
                        # 只有替代机器加工时间比等修复更快时才迁移
                        if best_time < current_time_on_broken + repair_time:
                            chromosome.ma[idx] = machines.index(best_alt)
                    # 否则保持原机器（等待修复）
                idx += 1

    def _reassign_agv_tasks(self, chromosome: Chromosome, event: DynamicEvent):
        """重新分配AGV任务（Level 1辅助）"""
        broken_agv = event.agv_id
        for idx in range(chromosome.n_total):
            if chromosome.agv_assign[idx] == broken_agv:
                # 分配给其他AGV
                available = [l for l in range(self.instance.num_agv) if l != broken_agv]
                if available:
                    chromosome.agv_assign[idx] = self.rng.choice(available)

    def _perturb_affected(self, chromosome: Chromosome, affected: set):
        """对受影响的工序做随机扰动"""
        idx = 0
        for i in range(self.instance.num_jobs):
            for j in range(self.instance.num_operations[i]):
                if (i, j) in affected:
                    if self.rng.random() < 0.5:
                        machines = self.instance.get_compatible_machines(i, j)
                        if machines:
                            chromosome.ma[idx] = self.rng.randint(0, len(machines))
                    if self.rng.random() < 0.3:
                        chromosome.agv_assign[idx] = self.rng.randint(0, self.instance.num_agv)
                    if self.rng.random() < 0.3:
                        chromosome.agv_speed[idx] = self.rng.randint(0, self.instance.num_speeds)
                idx += 1

    def _select_compromise(self, population: list) -> Chromosome:
        """选择折中解 式(113)"""
        objs = np.array([c.objectives.to_array() for c in population])
        ideal = objs.min(axis=0)
        nadir = objs.max(axis=0)
        range_vals = nadir - ideal
        range_vals[range_vals < 1e-10] = 1e-10
        normalized = (objs - ideal) / range_vals
        distances = np.sqrt(np.sum(normalized ** 2, axis=1))
        best_idx = int(np.argmin(distances))
        return population[best_idx]

    def _flat_to_op(self, inst, flat_idx):
        """扁平索引转(job, op)"""
        idx = 0
        for i in range(inst.num_jobs):
            for j in range(inst.num_operations[i]):
                if idx == flat_idx:
                    return (i, j)
                idx += 1
        return (0, 0)

    def get_response_summary(self) -> dict:
        """获取响应统计摘要"""
        if not self.response_log:
            return {}

        levels = [r['level'] for r in self.response_log]
        return {
            'total_events': len(self.response_log),
            'level1_count': levels.count(1),
            'level2_count': levels.count(2),
            'level3_count': levels.count(3),
            'avg_disruption': np.mean([r['disruption'] for r in self.response_log]),
            'avg_makespan_change': np.mean([
                r['makespan_after'] - r['makespan_before']
                for r in self.response_log
            ]),
        }
