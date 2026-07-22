"""多目标进化算法基线: NSGA-III (B2), SPEA2 (B3), MOEA/D (B4)

B2: 纯NSGA-III（随机算子选择）
B3: SPEA2（pymoo实现的选择逻辑近似）
B4: MOEA/D（基于切比雪夫分解）
"""

import numpy as np
from tqdm import tqdm
from src.problem.instance import FJSPAGVInstance
from src.algorithm.nsga3.parallel_eval import parallel_evaluate
from src.algorithm.nsga3.encoding import (
    Chromosome, random_chromosome, spt_chromosome,
    energy_chromosome, balance_chromosome
)
from src.algorithm.nsga3.crossover import CROSSOVER_OPERATORS
from src.algorithm.nsga3.mutation import MUTATION_OPERATORS
from src.algorithm.nsga3.selection import (
    generate_reference_points, nsga3_select, non_dominated_sort,
    compute_hypervolume
)


class _BaseEA:
    """进化算法公共基类"""

    def __init__(self, instance: FJSPAGVInstance,
                 pop_size: int = 200, max_gen: int = 500,
                 crossover_prob: float = 0.8, mutation_prob: float = 0.2,
                 seed: int = 42, verbose: bool = True):
        self.instance = instance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.history = {'hv': [], 'gen': []}

    def _initialize(self):
        population = []
        n = self.pop_size
        for _ in range(n // 4):
            population.append(spt_chromosome(self.instance, self.rng))
        for _ in range(n // 4):
            population.append(energy_chromosome(self.instance, self.rng))
        for _ in range(n // 4):
            population.append(balance_chromosome(self.instance, self.rng))
        while len(population) < n:
            population.append(random_chromosome(self.instance, self.rng))
        for c in population[:n]:
            _ = c.objectives
        return population[:n]

    def _generate_offspring(self, population):
        offspring = []
        for _ in range(self.pop_size // 2):
            p1 = self._tournament_select(population)
            p2 = self._tournament_select(population)

            if self.rng.random() < self.crossover_prob:
                op_id = self.rng.randint(0, 5)
                _, op_func = CROSSOVER_OPERATORS[op_id]
                c1, c2 = op_func(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if self.rng.random() < self.mutation_prob:
                op_id = self.rng.randint(5, 10)
                _, op_func = MUTATION_OPERATORS[op_id]
                if op_id >= 7:
                    c1 = op_func(c1, mutation_rate=0.1, rng=self.rng)
                    c2 = op_func(c2, mutation_rate=0.1, rng=self.rng)
                else:
                    c1 = op_func(c1, rng=self.rng)
                    c2 = op_func(c2, rng=self.rng)

            offspring.extend([c1, c2])

        for c in offspring:
            _ = c.objectives
        return offspring

    def _tournament_select(self, population, size=5):
        indices = self.rng.choice(len(population), min(size, len(population)), replace=False)
        candidates = [population[i] for i in indices]
        objs = np.array([c.objectives.to_array() for c in candidates])
        fronts = non_dominated_sort(objs)
        return candidates[fronts[0][0]]

    def _update_history(self, population):
        objs = np.array([c.objectives.to_array() for c in population])
        ref = objs.max(axis=0) * 1.1
        hv = compute_hypervolume(objs, ref)
        self.history['hv'].append(hv)
        self.history['gen'].append(len(self.history['gen']))
        return hv


class PureNSGA3(_BaseEA):
    """B2: 纯NSGA-III"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_points = generate_reference_points(3, 8)

    def run(self) -> list:
        population = self._initialize()
        archive = []

        iterator = range(self.max_gen)
        if self.verbose:
            iterator = tqdm(iterator, desc="NSGA-III")

        for gen in iterator:
            offspring = self._generate_offspring(population)
            combined = population + offspring
            objs = np.array([c.objectives.to_array() for c in combined])
            selected = nsga3_select(objs, self.pop_size, self.ref_points)
            population = [combined[i] for i in selected]

            # 档案
            all_sols = archive + population
            all_objs = np.array([c.objectives.to_array() for c in all_sols])
            fronts = non_dominated_sort(all_objs)
            archive = [all_sols[i] for i in fronts[0]][:100]

            hv = self._update_history(population)
            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'HV': f'{hv:.1f}'})

        return archive


class PureSPEA2(_BaseEA):
    """B3: SPEA2（强度Pareto进化算法）

    使用SPEA2的核心选择逻辑：强度值 + K近邻密度估计
    """

    def run(self) -> list:
        population = self._initialize()
        archive = []
        archive_size = self.pop_size

        iterator = range(self.max_gen)
        if self.verbose:
            iterator = tqdm(iterator, desc="SPEA2")

        for gen in iterator:
            offspring = self._generate_offspring(population)

            # SPEA2选择：强度值 + 密度估计
            combined = population + archive + offspring
            objs = np.array([c.objectives.to_array() for c in combined])

            # 计算强度值（被支配个体数）
            n = len(objs)
            strength = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                        strength[i] += 1

            # 原始适应度 = 被所有支配者的强度值之和
            raw_fitness = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                        raw_fitness[i] += strength[j]

            # K近邻密度（K = sqrt(N)）
            k = max(1, int(np.sqrt(n)))
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(objs[i] - objs[j])
                    dists[i, j] = dists[j, i] = d

            density = np.zeros(n)
            for i in range(n):
                sorted_dists = np.sort(dists[i])
                density[i] = 1.0 / (sorted_dists[k] + 2.0) if k < len(sorted_dists) else 0

            fitness = raw_fitness + density

            # 选择适应度最小的个体
            selected_idx = np.argsort(fitness)[:self.pop_size]
            population = [combined[i] for i in selected_idx]

            # 档案 = 非支配解
            fronts = non_dominated_sort(objs)
            archive = [combined[i] for i in fronts[0]][:archive_size]

            hv = self._update_history(population)
            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'HV': f'{hv:.1f}'})

        return archive


class PureMOEAD(_BaseEA):
    """B4: MOEA/D（基于切比雪夫分解的多目标进化）

    将多目标问题分解为多个单目标子问题，每个子问题由一个权重向量定义
    """

    def __init__(self, *args, neighborhood_size: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        # 生成均匀权重向量
        self.weights = generate_reference_points(3, 12)
        # 为每个权重向量定义邻域
        self.T = min(neighborhood_size, len(self.weights))
        dists = np.zeros((len(self.weights), len(self.weights)))
        for i in range(len(self.weights)):
            for j in range(len(self.weights)):
                dists[i, j] = np.linalg.norm(self.weights[i] - self.weights[j])
        self.neighbors = np.argsort(dists, axis=1)[:, :self.T]

    def run(self) -> list:
        population = self._initialize()
        # 截断或填充到权重向量数
        while len(population) < len(self.weights):
            population.append(random_chromosome(self.instance, self.rng))
            _ = population[-1].objectives
        population = population[:len(self.weights)]

        # 理想点
        objs = np.array([c.objectives.to_array() for c in population])
        z_ideal = objs.min(axis=0)

        archive = []

        iterator = range(self.max_gen)
        if self.verbose:
            iterator = tqdm(iterator, desc="MOEA/D")

        for gen in iterator:
            for i in range(len(self.weights)):
                # 从邻域中选择两个父代
                neighbor_indices = self.neighbors[i]
                idx1, idx2 = self.rng.choice(neighbor_indices, 2, replace=False)
                p1, p2 = population[idx1], population[idx2]

                # 交叉变异
                op_id = self.rng.randint(0, 5)
                _, op_func = CROSSOVER_OPERATORS[op_id]
                c1, _ = op_func(p1, p2, self.rng)

                if self.rng.random() < self.mutation_prob:
                    m_id = self.rng.randint(5, 10)
                    _, m_func = MUTATION_OPERATORS[m_id]
                    if m_id >= 7:
                        c1 = m_func(c1, mutation_rate=0.1, rng=self.rng)
                    else:
                        c1 = m_func(c1, rng=self.rng)

                _ = c1.objectives
                child_obj = c1.objectives.to_array()

                # 更新理想点
                z_ideal = np.minimum(z_ideal, child_obj)

                # 用切比雪夫方法更新邻域
                for j in neighbor_indices:
                    w = self.weights[j]
                    w_safe = np.maximum(w, 1e-6)
                    parent_obj = population[j].objectives.to_array()

                    child_tcheb = np.max(w_safe * np.abs(child_obj - z_ideal))
                    parent_tcheb = np.max(w_safe * np.abs(parent_obj - z_ideal))

                    if child_tcheb < parent_tcheb:
                        population[j] = c1

            # 档案
            all_objs = np.array([c.objectives.to_array() for c in population])
            fronts = non_dominated_sort(all_objs)
            archive = [population[i] for i in fronts[0]][:100]

            hv = self._update_history(population)
            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'HV': f'{hv:.1f}'})

        return archive
