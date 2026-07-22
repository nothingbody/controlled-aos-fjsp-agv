"""动态事件分级响应测试"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import generate_random_instance
from src.algorithm.nsga3.encoding import random_chromosome
from src.algorithm.nsga3.decoding import decode, evaluate
from src.environment.dynamic_events import EventGenerator, EventType
from src.algorithm.dynamic.disruption_evaluator import evaluate_disruption, find_affected_operations
from src.algorithm.dynamic.rescheduler import GradedRescheduler


def test_event_generation():
    """测试事件生成"""
    print("=" * 60)
    print("测试1：动态事件生成")
    inst = generate_random_instance(10, 6, 3, seed=42)

    gen = EventGenerator(
        inst,
        new_job_rate=0.3,
        machine_breakdown_rate=0.03,
        agv_breakdown_rate=0.02,
        seed=42
    )

    events = gen.generate_events(time_horizon=500, start_time=50)
    print(f"  生成事件数: {len(events)}")

    type_counts = {}
    for e in events:
        type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1

    for t, c in type_counts.items():
        print(f"    {t}: {c}个")

    print(f"  第一个事件: t={events[0].time:.1f}, type={events[0].event_type.value}")
    print(f"  最后一个事件: t={events[-1].time:.1f}, type={events[-1].event_type.value}")
    print("  PASSED")
    return inst, events


def test_disruption_evaluation(inst, events):
    """测试扰动评估"""
    print("\n" + "=" * 60)
    print("测试2：扰动程度评估")

    rng = np.random.RandomState(42)
    chrom = random_chromosome(inst, rng)
    schedule = decode(chrom)

    for e in events[:6]:
        D = evaluate_disruption(e, schedule, inst, current_time=e.time)
        direct, indirect = find_affected_operations(e, schedule, inst, e.time)
        print(f"  t={e.time:6.1f} | {e.event_type.value:20s} | "
              f"D={D:.3f} | 直接={len(direct)} | 间接={len(indirect)} | "
              f"级别={'L1' if D < 0.15 else 'L2' if D < 0.40 else 'L3'}")

    print("  PASSED")
    return chrom


def test_graded_response(inst, events, chrom):
    """测试分级响应"""
    print("\n" + "=" * 60)
    print("测试3：分级响应重调度")

    rescheduler = GradedRescheduler(
        inst,
        theta1=0.15, theta2=0.40,
        local_pop_size=20, local_max_gen=30,
        global_pop_size=30, global_max_gen=50,
        seed=42
    )

    current = chrom
    print(f"  初始 Cmax={current.objectives.makespan:.1f}, "
          f"TEC={current.objectives.total_energy:.1f}")

    # 处理前3个事件
    for i, e in enumerate(events[:3]):
        result = rescheduler.respond(
            event=e,
            current_schedule=current.schedule,
            current_chromosome=current,
            current_time=e.time
        )
        log = rescheduler.response_log[-1]
        print(f"  事件{i+1}: {e.event_type.value:20s} | "
              f"D={log['disruption']:.3f} | L{log['level']} | "
              f"Cmax: {log['makespan_before']:.1f} → {log['makespan_after']:.1f}")
        current = result

    # 统计
    summary = rescheduler.get_response_summary()
    print(f"\n  响应统计:")
    print(f"    总事件数: {summary['total_events']}")
    print(f"    L1/L2/L3: {summary['level1_count']}/{summary['level2_count']}/{summary['level3_count']}")
    print(f"    平均扰动: {summary['avg_disruption']:.3f}")
    print("  PASSED")


if __name__ == '__main__':
    print("GRL-EA 动态响应模块测试")
    print("=" * 60)

    inst, events = test_event_generation()
    chrom = test_disruption_evaluation(inst, events)
    test_graded_response(inst, events, chrom)

    print("\n" + "=" * 60)
    print("所有动态响应测试通过！")
    print("=" * 60)
