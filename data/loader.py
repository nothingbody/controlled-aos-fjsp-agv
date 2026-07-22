"""FJSP Benchmark 数据加载与解析

支持标准FJSP格式（Brandimarte、Hurink等）。
格式说明：
  第1行：num_jobs  num_machines  [avg_machines_per_op]
  后续每行表示一个工件：
    num_ops  k1  m1 p1 m2 p2 ...  k2  m1 p1 ...
    其中 ki 表示第i道工序可选的机器数，后跟 ki 对 (机器编号, 加工时间)
"""

import os
import hashlib
import numpy as np
from typing import Optional
from src.problem.instance import FJSPAGVInstance


def load_fjsp_file(filepath: str, num_agv: int = 3, seed: int = 42,
                   transport_time_ratio: float = 0.30) -> FJSPAGVInstance:
    """加载标准FJSP格式文件并生成AGV参数

    Args:
        filepath: .fjs文件路径
        num_agv: AGV数量
        seed: 随机种子（用于生成AGV参数）

    Returns:
        FJSPAGVInstance对象
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 解析第一行
    first_line = lines[0].split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    num_operations = []
    processing_times = {}
    compatible_machines = {}

    for i in range(num_jobs):
        line = lines[i + 1].split()
        idx = 0
        n_ops = int(line[idx])
        idx += 1
        num_operations.append(n_ops)

        for j in range(n_ops):
            num_compatible = int(line[idx])
            idx += 1
            machines = []
            for _ in range(num_compatible):
                machine = int(line[idx])  # SchedulingLab格式已是0-indexed
                proc_time = float(line[idx + 1])
                idx += 2
                machines.append(machine)
                processing_times[(i, j, machine)] = proc_time

            compatible_machines[(i, j)] = machines

    instance = FJSPAGVInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_agv=num_agv,
        num_operations=num_operations,
        processing_times=processing_times,
        compatible_machines=compatible_machines,
    )

    # 生成AGV相关参数
    instance.generate_agv_params(
        seed=seed,
        transport_time_ratio=transport_time_ratio,
    )

    return instance


def _derived_extension_seed(benchmark_dir: str, filename: str,
                            base_seed: int = 42) -> int:
    """Derive a stable, instance-specific seed for synthetic AGV parameters."""
    dataset = os.path.basename(os.path.normpath(benchmark_dir))
    token = f"{dataset}/{filename}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
    return int((int(base_seed) + offset) % (2 ** 32))


def load_benchmark_set(benchmark_dir: str, num_agv: int = 3,
                       extension_seed: int = 42,
                       transport_time_ratio: float = 0.30) -> list:
    """加载一个目录下的所有benchmark实例"""
    instances = []
    if not os.path.isdir(benchmark_dir):
        print(f"Warning: Directory {benchmark_dir} not found")
        return instances

    for filename in sorted(os.listdir(benchmark_dir)):
        if filename.endswith('.fjs'):
            filepath = os.path.join(benchmark_dir, filename)
            seed = _derived_extension_seed(
                benchmark_dir, filename, base_seed=extension_seed
            )
            instance = load_fjsp_file(
                filepath,
                num_agv=num_agv,
                seed=seed,
                transport_time_ratio=transport_time_ratio,
            )
            instance.extension_seed = seed
            instances.append((filename, instance))

    return instances


def generate_random_instance(
    num_jobs: int,
    num_machines: int,
    num_agv: int = 3,
    ops_range: tuple = (3, 10),
    compatible_range: tuple = (1, 4),
    pt_range: tuple = (1, 20),
    seed: int = 42
) -> FJSPAGVInstance:
    """随机生成FJSP-AGV实例

    Args:
        num_jobs: 工件数
        num_machines: 机器数
        num_agv: AGV数
        ops_range: 每个工件的工序数范围
        compatible_range: 每道工序的可选机器数范围
        pt_range: 加工时间范围
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)

    num_operations = []
    processing_times = {}
    compatible_machines = {}

    for i in range(num_jobs):
        n_ops = rng.randint(ops_range[0], ops_range[1] + 1)
        num_operations.append(n_ops)

        for j in range(n_ops):
            n_compat = min(
                rng.randint(compatible_range[0], compatible_range[1] + 1),
                num_machines
            )
            machines = sorted(rng.choice(num_machines, n_compat, replace=False).tolist())
            compatible_machines[(i, j)] = machines

            for u in machines:
                processing_times[(i, j, u)] = rng.randint(pt_range[0], pt_range[1] + 1)

    instance = FJSPAGVInstance(
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_agv=num_agv,
        num_operations=num_operations,
        processing_times=processing_times,
        compatible_machines=compatible_machines,
    )
    instance.generate_agv_params(seed=seed, transport_time_ratio=0.30)

    return instance


def download_brandimarte(target_dir: str = "data/benchmarks/brandimarte"):
    """提示用户下载Brandimarte数据集"""
    os.makedirs(target_dir, exist_ok=True)
    print(f"请从以下地址下载Brandimarte实例（Mk01-Mk10.fjs）并放入 {target_dir}/")
    print("  https://github.com/SchedulingLab/fjsp-instances")
    print("  或 https://github.com/PyJobShop/FJSPLIB")
