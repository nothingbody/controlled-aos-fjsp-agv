"""基础模块测试

验证：实例生成、编码、解码、评估、交叉、变异
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import generate_random_instance
from src.algorithm.nsga3.encoding import (
    random_chromosome, spt_chromosome, energy_chromosome, balance_chromosome
)
from src.algorithm.nsga3.decoding import decode, evaluate
from src.algorithm.nsga3.crossover import CROSSOVER_OPERATORS
from src.algorithm.nsga3.mutation import MUTATION_OPERATORS


def test_instance_generation():
    """测试随机实例生成"""
    print("=" * 60)
    print("测试1：随机实例生成")
    inst = generate_random_instance(
        num_jobs=10, num_machines=6, num_agv=3, seed=42
    )
    print(f"  {inst.summary()}")
    print(f"  工序数列表: {inst.num_operations}")
    print(f"  总工序数: {inst.total_operations}")
    print(f"  距离矩阵形状: {inst.distance_matrix.shape}")
    print(f"  机器加工功率: {inst.machine_proc_power.round(2)}")
    print(f"  AGV载货功率(v=1.5): {inst.agv_load_power(1.5):.3f} kW")
    print(f"  AGV运输能耗(d=30m, v=1.5): {inst.transport_energy(30, 1.5):.3f} kWh")
    print("  PASSED")
    return inst


def test_encoding(inst):
    """测试编码生成"""
    print("\n" + "=" * 60)
    print("测试2：染色体编码")
    rng = np.random.RandomState(42)

    chroms = {
        '随机': random_chromosome(inst, rng),
        'SPT': spt_chromosome(inst, rng),
        '能耗导向': energy_chromosome(inst, rng),
        '负载均衡': balance_chromosome(inst, rng),
    }

    for name, c in chroms.items():
        print(f"  {name}: OS长度={len(c.os)}, MA长度={len(c.ma)}, "
              f"AGV长度={len(c.agv_assign)}")
        # 验证OS中每个工件出现正确次数
        for i in range(inst.num_jobs):
            count = np.sum(c.os == i)
            assert count == inst.num_operations[i], \
                f"工件{i}出现{count}次，应为{inst.num_operations[i]}次"

    print("  所有编码合法性检查 PASSED")
    return chroms


def test_decoding(inst, chroms):
    """测试解码和评估"""
    print("\n" + "=" * 60)
    print("测试3：解码与评估")

    for name, c in chroms.items():
        schedule = decode(c)
        objectives = evaluate(c)
        print(f"  {name}:")
        print(f"    Cmax = {objectives.makespan:.2f}")
        print(f"    TEC  = {objectives.total_energy:.2f} kWh")
        print(f"    WB   = {objectives.workload_balance:.2f}")

        # 验证工艺约束
        for i in range(inst.num_jobs):
            for j in range(inst.num_operations[i] - 1):
                assert schedule.op_end[(i, j)] <= schedule.op_start[(i, j + 1)] + 1e-6, \
                    f"工艺约束违反: J{i}O{j} end={schedule.op_end[(i, j)]:.2f} > " \
                    f"J{i}O{j + 1} start={schedule.op_start[(i, j + 1)]:.2f}"

        # 验证非负时间
        for key in schedule.op_start:
            assert schedule.op_start[key] >= -1e-6, \
                f"工序{key}开始时间为负: {schedule.op_start[key]}"

    print("  所有约束检查 PASSED")


def test_crossover(inst):
    """测试交叉算子"""
    print("\n" + "=" * 60)
    print("测试4：交叉算子")
    rng = np.random.RandomState(42)

    p1 = random_chromosome(inst, rng)
    p2 = random_chromosome(inst, rng)

    for op_id, (name, op_func) in CROSSOVER_OPERATORS.items():
        c1, c2 = op_func(p1, p2, rng)
        # 验证合法性
        for c in [c1, c2]:
            for i in range(inst.num_jobs):
                count = np.sum(c.os == i)
                assert count == inst.num_operations[i], \
                    f"交叉{name}后工件{i}出现{count}次"
            # 验证可解码
            _ = decode(c)

        print(f"  C{op_id} ({name}): PASSED")


def test_mutation(inst):
    """测试变异算子"""
    print("\n" + "=" * 60)
    print("测试5：变异算子")
    rng = np.random.RandomState(42)

    parent = random_chromosome(inst, rng)

    for op_id, (name, op_func) in MUTATION_OPERATORS.items():
        if name in ('MachineReassign', 'AGVReassign', 'SpeedAdjust'):
            child = op_func(parent, mutation_rate=0.1, rng=rng)
        else:
            child = op_func(parent, rng=rng)

        # 验证合法性
        for i in range(inst.num_jobs):
            count = np.sum(child.os == i)
            assert count == inst.num_operations[i], \
                f"变异{name}后工件{i}出现{count}次"
        _ = decode(child)

        print(f"  M{op_id - 5} ({name}): PASSED")


def test_nsga3_selection():
    """测试NSGA-III选择"""
    print("\n" + "=" * 60)
    print("测试6：NSGA-III选择")
    from src.algorithm.nsga3.selection import (
        generate_reference_points, non_dominated_sort, nsga3_select
    )

    # 生成参考点
    ref_points = generate_reference_points(3, 4)
    print(f"  3目标4分割参考点数: {len(ref_points)}")

    ref_points_4d = generate_reference_points(4, 8)
    print(f"  4目标8分割参考点数: {len(ref_points_4d)}")

    # 非支配排序
    objectives = np.random.rand(50, 3)
    fronts = non_dominated_sort(objectives)
    total = sum(len(f) for f in fronts)
    print(f"  50个解非支配排序: {len(fronts)}层, 总计{total}个")
    assert total == 50

    # 环境选择
    ref_pts = generate_reference_points(3, 4)
    selected = nsga3_select(objectives, 20, ref_pts)
    print(f"  从50中选择20个: 选中{len(selected)}个")
    assert len(selected) == 20

    print("  PASSED")


if __name__ == '__main__':
    print("GRL-EA 基础模块测试")
    print("=" * 60)

    inst = test_instance_generation()
    chroms = test_encoding(inst)
    test_decoding(inst, chroms)
    test_crossover(inst)
    test_mutation(inst)
    test_nsga3_selection()

    print("\n" + "=" * 60)
    print("所有测试通过！基础模块可正常工作。")
    print("=" * 60)
