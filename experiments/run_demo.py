"""GRL-EA 端到端演示

在一个随机生成的小规模实例上运行GRL-EA，验证完整流程。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import generate_random_instance
from src.algorithm.grl_ea import GRLEA


def run_demo():
    print("=" * 60)
    print("GRL-EA 端到端演示")
    print("=" * 60)

    # 1. 生成小规模实例
    print("\n[1] 生成实例...")
    instance = generate_random_instance(
        num_jobs=6, num_machines=4, num_agv=2,
        ops_range=(2, 5), pt_range=(1, 15), seed=42
    )
    print(f"    {instance.summary()}")

    # 2. 运行GRL-EA（小规模，关闭GRL加速测试）
    print("\n[2] 运行GRL-EA（无GRL模式，验证EA框架）...")
    solver = GRLEA(
        instance=instance,
        pop_size=50,
        max_gen=30,
        use_grl=False,  # 先关闭GRL，纯EA验证
        seed=42,
        verbose=True,
    )
    archive = solver.run()

    # 3. 输出结果
    print(f"\n[3] 结果汇总")
    print(f"    Pareto解数量: {len(archive)}")

    print(f"\n    {'解编号':>6} | {'Cmax':>10} | {'TEC(kWh)':>10} | {'WB':>10}")
    print("    " + "-" * 50)
    for idx, c in enumerate(archive[:10]):  # 最多显示10个
        obj = c.objectives
        print(f"    {idx+1:>6} | {obj.makespan:>10.2f} | {obj.total_energy:>10.2f} | {obj.workload_balance:>10.2f}")

    # 4. 收敛曲线
    print(f"\n    HV收敛: 初始={solver.history['hv'][0]:.2f} → "
          f"最终={solver.history['hv'][-1]:.2f}")

    # 5. 运行GRL模式
    print("\n[4] 运行GRL-EA（GRL模式）...")
    solver_grl = GRLEA(
        instance=instance,
        pop_size=50,
        max_gen=30,
        use_grl=True,
        seed=42,
        verbose=True,
    )
    archive_grl = solver_grl.run()

    print(f"\n[5] 对比结果")
    print(f"    纯EA Pareto解数: {len(archive)}, 最终HV: {solver.history['hv'][-1]:.2f}")
    print(f"    GRL-EA Pareto解数: {len(archive_grl)}, 最终HV: {solver_grl.history['hv'][-1]:.2f}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    run_demo()
