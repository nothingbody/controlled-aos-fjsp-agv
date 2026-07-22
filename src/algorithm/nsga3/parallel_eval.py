"""并行评估器：利用多进程加速CPU密集的解码+评估过程

将种群的解码和评估分发到多个CPU核心并行处理。
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.encoding import Chromosome
from src.algorithm.nsga3.decoding import decode, evaluate


def parallel_evaluate(population: list, max_workers: int = None) -> None:
    """并行评估种群中所有个体

    使用多线程（因为Python对象共享内存，避免序列化开销）

    Args:
        population: Chromosome列表
        max_workers: 并行线程数（默认=CPU核心数）
    """
    # 筛选需要评估的个体
    to_eval = [c for c in population if c._objectives is None]

    if not to_eval:
        return

    if len(to_eval) <= 4:
        # 少量个体直接串行
        for c in to_eval:
            _ = c.objectives
        return

    if max_workers is None:
        import os
        max_workers = min(os.cpu_count() or 4, 8)

    def eval_single(chrom):
        _ = chrom.objectives
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(eval_single, to_eval))
