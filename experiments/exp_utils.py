"""实验公共工具：日志、增量保存、断点续跑"""

import os
import sys
import json
import logging
import time
from datetime import datetime
import pandas as pd


class ExperimentLogger:
    """实验日志器：同时输出到终端和文件"""

    def __init__(self, exp_name: str, log_dir: str = 'results/logs'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{exp_name}_{timestamp}.log')

        self.logger = logging.getLogger(exp_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # 文件handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S'))
        self.logger.addHandler(fh)

        # 终端handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

        self.log_file = log_file
        self.logger.info(f'Log file: {log_file}')

    def info(self, msg):
        self.logger.info(msg)

    def section(self, title):
        self.logger.info('=' * 70)
        self.logger.info(title)
        self.logger.info('=' * 70)


class IncrementalSaver:
    """增量结果保存器：每完成一条记录就追加保存，防止中途丢失"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.records = []

        # 如果文件已存在，加载已有记录（用于断点续跑）
        if os.path.exists(csv_path):
            try:
                existing = pd.read_csv(csv_path)
                self.records = existing.to_dict('records')
            except Exception:
                self.records = []

    def append(self, record: dict):
        """追加一条记录并立即写入文件"""
        self.records.append(record)
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)

    def has_result(self, **kwargs) -> bool:
        """检查是否已有某个条件的结果（用于断点续跑跳过）

        用法: saver.has_result(instance='Mk01.fjs', method='B1_Rules')
        """
        for record in self.records:
            match = all(record.get(k) == v for k, v in kwargs.items())
            if match:
                return True
        return False

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    @property
    def count(self):
        return len(self.records)


class Timer:
    """计时器"""

    def __init__(self):
        self.start_time = time.time()
        self.lap_time = self.start_time

    def elapsed(self) -> str:
        """总用时"""
        secs = time.time() - self.start_time
        return self._format(secs)

    def lap(self) -> str:
        """自上次lap以来的用时"""
        now = time.time()
        secs = now - self.lap_time
        self.lap_time = now
        return self._format(secs)

    @staticmethod
    def _format(secs):
        if secs < 60:
            return f'{secs:.1f}s'
        elif secs < 3600:
            return f'{secs/60:.1f}min'
        else:
            return f'{secs/3600:.1f}h'


def wilcoxon_test(data_a: list, data_b: list, alternative='two-sided') -> dict:
    """Wilcoxon秩和检验（用于比较两种方法的性能差异）

    Args:
        data_a: 方法A在各实例上的指标值列表
        data_b: 方法B在各实例上的指标值列表
        alternative: 'two-sided', 'less', 'greater'

    Returns:
        {'statistic': W, 'p_value': p, 'significant': bool}
    """
    from scipy import stats
    import numpy as np

    a = np.array(data_a, dtype=float)
    b = np.array(data_b, dtype=float)

    # 移除相等的对（Wilcoxon要求差值非零）
    diff = a - b
    nonzero = diff != 0
    if nonzero.sum() < 3:
        return {'statistic': float('nan'), 'p_value': 1.0, 'significant': False}

    try:
        stat, p = stats.wilcoxon(a[nonzero], b[nonzero], alternative=alternative)
    except Exception:
        return {'statistic': float('nan'), 'p_value': 1.0, 'significant': False}

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': p < 0.05,
    }


def friedman_test(data_matrix) -> dict:
    """Friedman检验（用于多方法多实例的总体比较）

    Args:
        data_matrix: shape (n_instances, n_methods) 的指标值矩阵

    Returns:
        {'statistic': chi2, 'p_value': p, 'significant': bool}
    """
    from scipy import stats
    import numpy as np

    data = np.array(data_matrix, dtype=float)
    try:
        stat, p = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
    except Exception:
        return {'statistic': float('nan'), 'p_value': 1.0, 'significant': False}

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'significant': p < 0.05,
    }


def estimate_remaining(done: int, total: int, elapsed_secs: float) -> str:
    """估算剩余时间"""
    if done <= 0:
        return '?'
    rate = elapsed_secs / done
    remaining = rate * (total - done)
    if remaining < 60:
        return f'{remaining:.0f}s'
    elif remaining < 3600:
        return f'{remaining/60:.0f}min'
    else:
        return f'{remaining/3600:.1f}h'


# ====== 原始 Pareto 前沿保存 ======
import pickle
from pathlib import Path


def save_pareto_front(objs, method, instance, seed, exp_name):
    """保存单次运行的原始非支配解 F 数组。

    用途：后处理重算 HV/IGD/Spread 时需要原始前沿。
    路径：results/pareto_fronts/{exp_name}/{instance}_{method}_seed{seed}.pkl
    """
    out_dir = Path(f'results/pareto_fronts/{exp_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_clean = str(instance).replace('.fjs', '').replace('/', '_').replace('\\', '_')
    fname = out_dir / f'{inst_clean}_{method}_seed{seed}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump({
            'objectives': objs,
            'method': method,
            'instance': str(instance),
            'seed': seed,
        }, f)
