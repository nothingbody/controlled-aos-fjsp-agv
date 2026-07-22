"""动态事件生成与注入

对应论文第3章 3.6节 和第6章
支持三类事件：新订单到达、机器故障、AGV故障
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List
from src.problem.instance import FJSPAGVInstance


class EventType(Enum):
    NEW_JOB = "new_job"
    MACHINE_BREAKDOWN = "machine_breakdown"
    AGV_BREAKDOWN = "agv_breakdown"


@dataclass
class DynamicEvent:
    """动态事件"""
    event_type: EventType
    time: float               # 事件发生时刻
    # 新订单参数
    new_job_id: int = -1
    new_num_ops: int = 0
    new_processing_times: dict = None  # {(job, op, machine): time}
    new_compatible_machines: dict = None  # {(job, op): [machines]}
    # 机器故障参数
    machine_id: int = -1
    repair_time: float = 0.0
    # AGV故障参数
    agv_id: int = -1
    agv_repair_time: float = 0.0


class EventGenerator:
    """动态事件生成器

    基于泊松过程和指数分布生成三类动态事件
    """

    def __init__(self, instance: FJSPAGVInstance,
                 new_job_rate: float = 0.2,
                 machine_breakdown_rate: float = 0.02,
                 agv_breakdown_rate: float = 0.01,
                 machine_repair_range: tuple = (5, 20),
                 agv_repair_range: tuple = (3, 15),
                 new_job_ops_range: tuple = (2, 5),
                 new_job_pt_range: tuple = (1, 15),
                 seed: int = 42):
        self.instance = instance
        self.new_job_rate = new_job_rate
        self.machine_breakdown_rate = machine_breakdown_rate
        self.agv_breakdown_rate = agv_breakdown_rate
        self.machine_repair_range = machine_repair_range
        self.agv_repair_range = agv_repair_range
        self.new_job_ops_range = new_job_ops_range
        self.new_job_pt_range = new_job_pt_range
        self.rng = np.random.RandomState(seed)
        self._next_job_id = instance.num_jobs

    def generate_events(self, time_horizon: float,
                        start_time: float = 0.0) -> List[DynamicEvent]:
        """在时间范围内生成所有动态事件

        Args:
            time_horizon: 时间范围上限
            start_time: 开始时间（避免在调度初期就触发事件）

        Returns:
            按时间排序的事件列表
        """
        events = []

        # 新订单到达（泊松过程）
        if self.new_job_rate > 0:
            t = start_time
            while True:
                interval = self.rng.exponential(1.0 / self.new_job_rate)
                t += interval
                if t >= time_horizon:
                    break
                events.append(self._generate_new_job_event(t))

        # 机器故障（各机器独立的指数分布）
        if self.machine_breakdown_rate > 0:
            for u in range(self.instance.num_machines):
                t = start_time
                while True:
                    interval = self.rng.exponential(1.0 / self.machine_breakdown_rate)
                    t += interval
                    if t >= time_horizon:
                        break
                    repair = self.rng.uniform(*self.machine_repair_range)
                    events.append(DynamicEvent(
                        event_type=EventType.MACHINE_BREAKDOWN,
                        time=t,
                        machine_id=u,
                        repair_time=repair,
                    ))

        # AGV故障
        if self.agv_breakdown_rate > 0:
            for l in range(self.instance.num_agv):
                t = start_time
                while True:
                    interval = self.rng.exponential(1.0 / self.agv_breakdown_rate)
                    t += interval
                    if t >= time_horizon:
                        break
                    repair = self.rng.uniform(*self.agv_repair_range)
                    events.append(DynamicEvent(
                        event_type=EventType.AGV_BREAKDOWN,
                        time=t,
                        agv_id=l,
                        agv_repair_time=repair,
                    ))

        events.sort(key=lambda e: e.time)
        return events

    def _generate_new_job_event(self, time: float) -> DynamicEvent:
        """生成新订单到达事件"""
        job_id = self._next_job_id
        self._next_job_id += 1

        num_ops = self.rng.randint(*self.new_job_ops_range)
        processing_times = {}
        compatible_machines = {}

        for j in range(num_ops):
            n_compat = self.rng.randint(1, min(4, self.instance.num_machines + 1))
            machines = sorted(
                self.rng.choice(self.instance.num_machines, n_compat, replace=False).tolist()
            )
            compatible_machines[(job_id, j)] = machines
            for u in machines:
                processing_times[(job_id, j, u)] = self.rng.randint(*self.new_job_pt_range)

        return DynamicEvent(
            event_type=EventType.NEW_JOB,
            time=time,
            new_job_id=job_id,
            new_num_ops=num_ops,
            new_processing_times=processing_times,
            new_compatible_machines=compatible_machines,
        )
