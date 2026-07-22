"""右移重调度基线 (用于动态场景对比)

遇到动态事件时，仅将受影响工序向后平移，不做重新优化。
"""

import numpy as np
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.encoding import Chromosome
from src.algorithm.nsga3.decoding import Schedule
from src.environment.dynamic_events import DynamicEvent, EventType
from src.algorithm.dynamic.disruption_evaluator import find_affected_operations


class RightShiftRescheduler:
    """右移重调度"""

    def __init__(self, instance: FJSPAGVInstance, seed: int = 42):
        self.instance = instance
        self.rng = np.random.RandomState(seed)
        self.response_log = []

    def respond(self, event: DynamicEvent, current_schedule: Schedule,
                current_chromosome: Chromosome, current_time: float) -> Chromosome:
        """右移响应：受影响工序及其后续全部右移"""
        child = current_chromosome.copy()

        if event.event_type == EventType.NEW_JOB:
            # 新工件追加到末尾
            new_entries = np.full(event.new_num_ops, event.new_job_id)
            child.os = np.concatenate([child.os, new_entries])
            new_ma = np.zeros(event.new_num_ops, dtype=int)
            child.ma = np.concatenate([child.ma, new_ma])
            child.agv_assign = np.concatenate([
                child.agv_assign,
                self.rng.randint(0, self.instance.num_agv, size=event.new_num_ops)
            ])
            child.agv_speed = np.concatenate([
                child.agv_speed,
                np.ones(event.new_num_ops, dtype=int)
            ])

        elif event.event_type == EventType.MACHINE_BREAKDOWN:
            # 不做任何调整，等机器修好后继续（相当于延迟）
            pass

        elif event.event_type == EventType.AGV_BREAKDOWN:
            # 不做任何调整
            pass

        child.invalidate()

        cmax_before = current_chromosome.objectives.makespan
        cmax_after = child.objectives.makespan

        self.response_log.append({
            'time': current_time,
            'event_type': event.event_type.value,
            'level': 0,
            'makespan_before': cmax_before,
            'makespan_after': cmax_after,
        })

        return child

    def get_response_summary(self) -> dict:
        if not self.response_log:
            return {}
        return {
            'total_events': len(self.response_log),
            'avg_makespan_change': np.mean([
                r['makespan_after'] - r['makespan_before']
                for r in self.response_log
            ]),
        }


class FullRescheduler:
    """完全重调度基线：每次事件触发完整EA重新求解"""

    def __init__(self, instance: FJSPAGVInstance,
                 pop_size: int = 100, max_gen: int = 200, seed: int = 42):
        self.instance = instance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.seed = seed
        self.response_log = []

    def respond(self, event: DynamicEvent, current_schedule: Schedule,
                current_chromosome: Chromosome, current_time: float) -> Chromosome:
        """完全重调度：每次都运行完整NSGA-III"""
        from src.baselines.pure_nsga3 import PureNSGA3

        # 处理新工件
        if event.event_type == EventType.NEW_JOB:
            self.instance.num_jobs += 1
            self.instance.num_operations.append(event.new_num_ops)
            if event.new_processing_times:
                self.instance.processing_times.update(event.new_processing_times)
            if event.new_compatible_machines:
                self.instance.compatible_machines.update(event.new_compatible_machines)

        solver = PureNSGA3(
            self.instance,
            pop_size=self.pop_size,
            max_gen=self.max_gen,
            seed=self.seed,
            verbose=False
        )
        archive = solver.run()

        best = min(archive, key=lambda c: c.objectives.makespan)

        cmax_before = current_chromosome.objectives.makespan
        cmax_after = best.objectives.makespan

        self.response_log.append({
            'time': current_time,
            'event_type': event.event_type.value,
            'level': 3,
            'makespan_before': cmax_before,
            'makespan_after': cmax_after,
        })

        return best

    def get_response_summary(self) -> dict:
        if not self.response_log:
            return {}
        return {
            'total_events': len(self.response_log),
            'avg_makespan_change': np.mean([
                r['makespan_after'] - r['makespan_before']
                for r in self.response_log
            ]),
        }
