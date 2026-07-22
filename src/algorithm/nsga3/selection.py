"""NSGA-III 环境选择

对应论文第5章 5.7节 Algorithm 3
"""

import numpy as np
from typing import List


def generate_reference_points(num_objectives: int, divisions: int) -> np.ndarray:
    """Das-Dennis方法生成参考点

    Args:
        num_objectives: 目标数 M
        divisions: 分割数 p

    Returns:
        参考点矩阵 (H, M)
    """
    ref_points = []
    _das_dennis_recursive(num_objectives, divisions, [],
                          divisions, ref_points)
    return np.array(ref_points)


def _das_dennis_recursive(M, p, current, remaining, result):
    if len(current) == M - 1:
        current.append(remaining / p)
        result.append(current[:])
        current.pop()
        return
    for i in range(remaining + 1):
        current.append(i / p)
        _das_dennis_recursive(M, p, current, remaining - i, result)
        current.pop()


def non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """快速非支配排序

    Args:
        objectives: 目标值矩阵 (N, M)，越小越好

    Returns:
        各层的个体索引列表 [[front1_indices], [front2_indices], ...]
    """
    n = len(objectives)
    domination_count = np.zeros(n, dtype=int)
    dominated_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(objectives[j], objectives[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    # 移除最后一个空层
    if not fronts[-1]:
        fronts.pop()

    return fronts


def _dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """判断a是否支配b（所有目标不差且至少一个更好）"""
    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)


def _niching_select(last_front: List[int], last_associations: np.ndarray,
                    last_distances: np.ndarray, niche_count: np.ndarray,
                    remaining: int, rng=None) -> List[int]:
    """Select solutions from a split front using NSGA-III niching.

    Only reference directions that still have an associated candidate are
    eligible.  Following Deb and Jain's niching rule, the closest candidate is
    selected for an empty niche; otherwise a candidate is selected at random.

    ``rng`` may be ``numpy.random``, a ``RandomState``, or a ``Generator``.
    Keeping it optional preserves the historical, ``np.random.seed``-based
    behavior of callers while allowing deterministic unit tests and isolated
    random streams.
    """
    rng = np.random if rng is None else rng
    last_front = list(last_front)
    associations = np.asarray(last_associations, dtype=int)
    distances = np.asarray(last_distances, dtype=float)

    if len(last_front) != len(associations) or len(last_front) != len(distances):
        raise ValueError("Split-front indices, associations, and distances must align")
    if remaining <= 0 or not last_front:
        return []

    active = np.ones(len(last_front), dtype=bool)
    chosen = []

    while len(chosen) < remaining and np.any(active):
        # Directions without an active split-front candidate must not compete
        # for the minimum niche count.
        eligible_refs = np.unique(associations[active])
        eligible_counts = niche_count[eligible_refs]
        min_count = eligible_counts.min()
        min_refs = eligible_refs[eligible_counts == min_count]
        chosen_ref = int(rng.choice(min_refs))

        candidates = np.flatnonzero(active & (associations == chosen_ref))
        if niche_count[chosen_ref] == 0:
            # np.argmin is deterministic for equal distances.
            pick = int(candidates[np.argmin(distances[candidates])])
        else:
            pick = int(rng.choice(candidates))

        chosen.append(last_front[pick])
        active[pick] = False
        niche_count[chosen_ref] += 1

    return chosen


def _normalize_nsga3_objectives(objectives: np.ndarray,
                                extreme_indices=None) -> np.ndarray:
    """Normalize using Deb--Jain ASF extreme points and intercepts.

    Degenerate or invalid hyperplanes fall back to the worst point of the
    nominated nondominated front, with a final population-range safeguard for
    constant objectives.
    """
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2 or len(objectives) == 0:
        raise ValueError("objectives must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(objectives)):
        raise ValueError("objectives must contain only finite values")

    ideal = objectives.min(axis=0)
    worst_population = objectives.max(axis=0)
    translated = objectives - ideal

    if extreme_indices is None:
        extreme_indices = np.arange(len(objectives), dtype=int)
    else:
        extreme_indices = np.asarray(extreme_indices, dtype=int)
    if extreme_indices.ndim != 1 or len(extreme_indices) == 0:
        raise ValueError("extreme_indices must identify at least one solution")
    if np.any(extreme_indices < 0) or np.any(extreme_indices >= len(objectives)):
        raise ValueError("extreme_indices contain an out-of-range index")

    front_translated = translated[extreme_indices].copy()
    front_translated[front_translated < 1e-12] = 0.0
    n_objectives = objectives.shape[1]
    weights = np.eye(n_objectives)
    weights[weights == 0.0] = 1e6
    asf = np.max(
        front_translated[None, :, :] * weights[:, None, :],
        axis=2,
    )
    extreme_points = front_translated[np.argmin(asf, axis=1)]

    worst_front = objectives[extreme_indices].max(axis=0)
    try:
        plane = np.linalg.solve(extreme_points, np.ones(n_objectives))
        with np.errstate(divide="ignore", invalid="ignore"):
            intercepts = 1.0 / plane
        if (
            not np.all(np.isfinite(intercepts))
            or np.any(intercepts <= 1e-12)
            or not np.allclose(extreme_points @ plane, 1.0)
        ):
            raise np.linalg.LinAlgError("invalid NSGA-III hyperplane")
        nadir = np.minimum(ideal + intercepts, worst_population)
    except (np.linalg.LinAlgError, FloatingPointError):
        nadir = worst_front.copy()

    spans = nadir - ideal
    small = spans <= 1e-12
    spans[small] = (worst_population - ideal)[small]
    spans = np.where(spans > 1e-12, spans, 1.0)
    return translated / spans


def nsga3_select(objectives: np.ndarray, pop_size: int,
                 ref_points: np.ndarray, rng=None) -> np.ndarray:
    """NSGA-III环境选择

    Args:
        objectives: 合并种群的目标值矩阵 (N, M)
        pop_size: 目标种群大小
        ref_points: 参考点矩阵 (H, M)
        rng: 可选的 NumPy 随机数生成器；默认使用全局 ``np.random``

    Returns:
        选中个体的索引数组
    """
    objectives = np.asarray(objectives, dtype=float)
    ref_points = np.asarray(ref_points, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("objectives must be a two-dimensional array")
    if ref_points.ndim != 2 or ref_points.shape[1] != objectives.shape[1]:
        raise ValueError("ref_points must have the same objective dimension")
    if len(ref_points) == 0:
        raise ValueError("at least one reference point is required")
    if not np.all(np.isfinite(objectives)) or not np.all(np.isfinite(ref_points)):
        raise ValueError("objectives and ref_points must contain only finite values")
    ref_norms = np.linalg.norm(ref_points, axis=1)
    if np.any(ref_norms <= np.finfo(float).eps):
        raise ValueError("reference directions must be non-zero")
    if pop_size <= 0 or len(objectives) == 0:
        return np.array([], dtype=int)

    fronts = non_dominated_sort(objectives)

    selected = []
    front_idx = 0

    # 逐层加入，直到超过pop_size
    while front_idx < len(fronts) and len(selected) + len(fronts[front_idx]) <= pop_size:
        selected.extend(fronts[front_idx])
        front_idx += 1

    if len(selected) == pop_size:
        return np.array(selected)

    if front_idx >= len(fronts):
        return np.array(selected[:pop_size])

    # 从最后一个需要截断的层中选择
    remaining = pop_size - len(selected)
    last_front = fronts[front_idx]

    if remaining <= 0:
        return np.array(selected[:pop_size])

    # 目标值归一化
    all_indices = selected + last_front
    all_objs = objectives[all_indices]

    position = {
        original_idx: local_idx
        for local_idx, original_idx in enumerate(all_indices)
    }
    first_front_positions = [position[idx] for idx in fronts[0] if idx in position]
    normalized = _normalize_nsga3_objectives(
        all_objs,
        extreme_indices=first_front_positions,
    )

    # 计算每个个体到每个参考点的垂直距离
    n_selected = len(selected)
    # 关联计数（已选中个体关联到各参考点的数量）
    niche_count = np.zeros(len(ref_points), dtype=int)

    # 计算所有个体到参考点的距离和关联
    associations = np.zeros(len(all_indices), dtype=int)
    distances = np.zeros(len(all_indices))

    # Normalize directions once; perpendicular projection is then stable and
    # avoids repeatedly dividing by very small squared norms.
    unit_refs = ref_points / ref_norms[:, None]
    for i in range(len(all_indices)):
        projections = (unit_refs @ normalized[i])[:, None] * unit_refs
        ref_distances = np.linalg.norm(normalized[i] - projections, axis=1)
        min_ref = int(np.argmin(ref_distances))
        associations[i] = min_ref
        distances[i] = ref_distances[min_ref]

    # 统计已选中个体的关联计数
    for i in range(n_selected):
        niche_count[associations[i]] += 1

    chosen_from_last = _niching_select(
        last_front,
        associations[n_selected:],
        distances[n_selected:],
        niche_count,
        remaining,
        rng=rng,
    )

    selected.extend(chosen_from_last)
    return np.array(selected[:pop_size])


def compute_hypervolume(objectives: np.ndarray, ref_point: np.ndarray) -> float:
    """计算超体积指标（2D精确计算，高维近似）

    Args:
        objectives: 非支配解集的目标值 (N, M)
        ref_point: 参考点

    Returns:
        超体积值
    """
    if len(objectives) == 0:
        return 0.0

    M = objectives.shape[1]

    # 优先使用pymoo的精确HV（如果可用）
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=ref_point)
        return float(ind(objectives))
    except (ImportError, Exception):
        pass

    if M == 2:
        return _hv_2d(objectives, ref_point)
    else:
        return _hv_monte_carlo(objectives, ref_point, n_samples=10000)


def _hv_2d(objectives: np.ndarray, ref_point: np.ndarray) -> float:
    """2D精确超体积计算"""
    # 按第一个目标排序
    sorted_idx = np.argsort(objectives[:, 0])
    sorted_objs = objectives[sorted_idx]

    hv = 0.0
    prev_y = ref_point[1]

    for i in range(len(sorted_objs)):
        if sorted_objs[i, 0] < ref_point[0] and sorted_objs[i, 1] < ref_point[1]:
            x_width = ref_point[0] - sorted_objs[i, 0]
            if i < len(sorted_objs) - 1:
                x_width = min(x_width, sorted_objs[i + 1, 0] - sorted_objs[i, 0])
            y_height = prev_y - sorted_objs[i, 1]
            if y_height > 0:
                hv += x_width * y_height
                prev_y = sorted_objs[i, 1]

    return hv


def _hv_monte_carlo(objectives: np.ndarray, ref_point: np.ndarray,
                    n_samples: int = 10000) -> float:
    """蒙特卡洛近似超体积"""
    ideal = objectives.min(axis=0)
    volume = np.prod(ref_point - ideal)

    if volume <= 0:
        return 0.0

    samples = np.random.uniform(ideal, ref_point, size=(n_samples, len(ref_point)))

    dominated_count = 0
    for s in samples:
        if np.any(np.all(objectives <= s, axis=1)):
            dominated_count += 1

    return volume * dominated_count / n_samples
