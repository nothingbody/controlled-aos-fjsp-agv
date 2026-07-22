"""RL-EA混合基线方法: HRLMA (B6) 和 KEARL (B7)

B6: HRLMA近似 — RL自适应调整交叉/变异概率（浅层辅助）
B7: KEARL近似 — 知识引导初始化 + RL自适应调参 + 知识引导VNS

这两个方法是对原始论文核心思想的忠实近似实现。
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
from src.algorithm.grl.critical_path import find_critical_path


class HRLMA:
    """B6: HRLMA近似（Hybrid RL-assisted Memetic Algorithm）

    核心思想：用Q-learning自适应调整交叉概率和变异概率
    对应论文EAAI 2026的简化实现
    """

    def __init__(self, instance: FJSPAGVInstance,
                 pop_size: int = 200, max_gen: int = 500,
                 seed: int = 42, verbose: bool = True):
        self.instance = instance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.ref_points = generate_reference_points(3, 8)
        self.history = {'hv': [], 'gen': []}

        # Q-learning参数调整
        self.pc = 0.8   # 交叉概率
        self.pm = 0.2   # 变异概率
        # Q表：状态(改善/停滞) × 动作(增加pc/减少pc/增加pm/减少pm)
        self.q_table = np.zeros((2, 4))
        self.lr_q = 0.1
        self.gamma_q = 0.9
        self.epsilon_q = 0.2

    def run(self) -> list:
        population = self._initialize()
        archive = []

        pop_objs = np.array([c.objectives.to_array() for c in population])
        ref = pop_objs.max(axis=0) * 1.1
        prev_hv = compute_hypervolume(pop_objs, ref)

        iterator = range(self.max_gen)
        if self.verbose:
            iterator = tqdm(iterator, desc="HRLMA")

        for gen in iterator:
            # Q-learning选择参数调整动作
            if gen == 0 or len(self.history['hv']) < 2:
                state = 0
            else:
                state = 0 if self.history['hv'][-1] > self.history['hv'][-2] * 1.001 else 1
            action = self._ql_select(state)
            self._apply_param_action(action)

            # 进化搜索
            offspring = self._generate_offspring(population)

            # VNS局部搜索（Memetic部分）
            for i in range(min(10, len(offspring))):
                offspring[i] = self._vns_local_search(offspring[i])

            combined = population + offspring
            objs = np.array([c.objectives.to_array() for c in combined])
            selected = nsga3_select(objs, self.pop_size, self.ref_points)
            population = [combined[i] for i in selected]

            # 更新档案
            all_objs = np.array([c.objectives.to_array() for c in population + archive])
            fronts = non_dominated_sort(all_objs)
            archive = [([*population, *archive])[i] for i in fronts[0]][:100]

            # 计算HV并更新Q表
            pop_objs = np.array([c.objectives.to_array() for c in population])
            new_hv = compute_hypervolume(pop_objs, ref)
            reward = 1.0 if new_hv > prev_hv * 1.001 else -0.1
            next_state = 0 if new_hv > prev_hv * 1.001 else 1
            self.q_table[state, action] += self.lr_q * (
                reward + self.gamma_q * self.q_table[next_state].max() - self.q_table[state, action]
            )

            self.history['hv'].append(new_hv)
            self.history['gen'].append(gen)
            prev_hv = new_hv

            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'HV': f'{new_hv:.1f}', 'pc': f'{self.pc:.2f}', 'pm': f'{self.pm:.2f}'})

        return archive

    def _ql_select(self, state):
        if self.rng.random() < self.epsilon_q:
            return self.rng.randint(0, 4)
        return int(np.argmax(self.q_table[state]))

    def _apply_param_action(self, action):
        delta = 0.05
        if action == 0:
            self.pc = min(0.95, self.pc + delta)
        elif action == 1:
            self.pc = max(0.4, self.pc - delta)
        elif action == 2:
            self.pm = min(0.5, self.pm + delta)
        elif action == 3:
            self.pm = max(0.05, self.pm - delta)

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
            if self.rng.random() < self.pc:
                op_id = self.rng.randint(0, 5)
                _, op_func = CROSSOVER_OPERATORS[op_id]
                c1, c2 = op_func(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if self.rng.random() < self.pm:
                op_id = self.rng.randint(5, 10)
                _, op_func = MUTATION_OPERATORS[op_id]
                if op_id >= 7:
                    c1 = op_func(c1, mutation_rate=0.15, rng=self.rng)
                    c2 = op_func(c2, mutation_rate=0.15, rng=self.rng)
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

    def _vns_local_search(self, chromosome: Chromosome) -> Chromosome:
        """变邻域搜索（3种邻域）"""
        best = chromosome
        best_obj = best.objectives.to_array()

        for neighborhood in range(3):
            candidate = best.copy()
            if neighborhood == 0:
                # 关键路径邻域：交换关键路径上两个相邻工序
                cp = find_critical_path(candidate.schedule, self.instance)
                if len(cp) >= 2:
                    idx = self.rng.randint(0, len(cp) - 1)
                    op1, op2 = cp[idx], cp[idx + 1]
                    if op1[0] != op2[0]:  # 不同工件才能交换
                        pos1 = self._find_in_os(candidate.os, op1[0], op1[1])
                        pos2 = self._find_in_os(candidate.os, op2[0], op2[1])
                        if pos1 is not None and pos2 is not None:
                            candidate.os[pos1], candidate.os[pos2] = candidate.os[pos2], candidate.os[pos1]
            elif neighborhood == 1:
                # 机器重分配邻域
                flat = self.rng.randint(0, candidate.n_total)
                i, j = self._flat_to_op(flat)
                if i is not None:
                    machines = self.instance.get_compatible_machines(i, j)
                    if len(machines) > 1:
                        candidate.ma[flat] = self.rng.randint(0, len(machines))
            else:
                # AGV+速度邻域
                flat = self.rng.randint(0, candidate.n_total)
                candidate.agv_assign[flat] = self.rng.randint(0, self.instance.num_agv)
                candidate.agv_speed[flat] = self.rng.randint(0, self.instance.num_speeds)

            candidate.invalidate()
            cand_obj = candidate.objectives.to_array()
            if np.any(cand_obj < best_obj):
                best = candidate
                best_obj = cand_obj

        return best

    def _find_in_os(self, os_seq, job, op):
        count = 0
        for pos in range(len(os_seq)):
            if os_seq[pos] == job:
                if count == op:
                    return pos
                count += 1
        return None

    def _flat_to_op(self, flat_idx):
        idx = 0
        for i in range(self.instance.num_jobs):
            for j in range(self.instance.num_operations[i]):
                if idx == flat_idx:
                    return i, j
                idx += 1
        return None, None


class KEARL:
    """B7: KEARL近似（Knowledge-guided EA with RL）

    核心思想：
    1. 4种知识引导初始化
    2. 协作式RL自适应调参
    3. 知识引导变邻域搜索
    对应Swarm&EC 2025的简化实现
    """

    def __init__(self, instance: FJSPAGVInstance,
                 pop_size: int = 200, max_gen: int = 500,
                 seed: int = 42, verbose: bool = True):
        self.instance = instance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.ref_points = generate_reference_points(3, 8)
        self.history = {'hv': [], 'gen': []}

        # 双Q-learning（协作式）
        self.pc = 0.8
        self.pm = 0.2
        self.q1 = np.zeros((3, 4))  # Q-learning agent 1
        self.q2 = np.zeros((3, 4))  # Q-learning agent 2（SARSA）
        self.lr_q = 0.1

    def run(self) -> list:
        # 知识引导初始化（4种策略）
        population = self._knowledge_guided_init()
        archive = []

        pop_objs = np.array([c.objectives.to_array() for c in population])
        ref = pop_objs.max(axis=0) * 1.1
        prev_hv = compute_hypervolume(pop_objs, ref)

        iterator = range(self.max_gen)
        if self.verbose:
            iterator = tqdm(iterator, desc="KEARL")

        prev_state = 0
        prev_action = 0

        for gen in iterator:
            # 状态：根据3个目标的改善情况
            if gen == 0:
                state = 0
            else:
                hv_ratio = self.history['hv'][-1] / max(prev_hv, 1e-10)
                state = 0 if hv_ratio > 1.005 else (1 if hv_ratio > 0.999 else 2)

            # 协作式RL选择动作
            action = self._cooperative_rl_select(state)
            self._apply_param_action(action)

            # 进化
            offspring = self._generate_offspring(population)

            # 知识引导VNS
            for i in range(min(15, len(offspring))):
                offspring[i] = self._knowledge_guided_vns(offspring[i])

            combined = population + offspring
            objs = np.array([c.objectives.to_array() for c in combined])
            selected = nsga3_select(objs, self.pop_size, self.ref_points)
            population = [combined[i] for i in selected]

            all_sols = archive + population
            all_objs = np.array([c.objectives.to_array() for c in all_sols])
            fronts = non_dominated_sort(all_objs)
            archive = [all_sols[i] for i in fronts[0]][:100]

            pop_objs = np.array([c.objectives.to_array() for c in population])
            new_hv = compute_hypervolume(pop_objs, ref)

            # 更新双Q表
            reward = 1.0 if new_hv > prev_hv * 1.001 else (-0.5 if new_hv < prev_hv * 0.999 else 0.0)
            next_state = 0 if new_hv > prev_hv * 1.005 else (1 if new_hv > prev_hv * 0.999 else 2)

            # Q-learning更新
            self.q1[state, action] += self.lr_q * (
                reward + 0.9 * self.q1[next_state].max() - self.q1[state, action]
            )
            # SARSA更新
            next_action = self._cooperative_rl_select(next_state)
            self.q2[prev_state, prev_action] += self.lr_q * (
                reward + 0.9 * self.q2[state, action] - self.q2[prev_state, prev_action]
            )

            prev_state, prev_action = state, action
            self.history['hv'].append(new_hv)
            self.history['gen'].append(gen)
            prev_hv = new_hv

            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'HV': f'{new_hv:.1f}'})

        return archive

    def _cooperative_rl_select(self, state):
        """协作式选择：Q1和Q2的平均Q值"""
        if self.rng.random() < 0.15:
            return self.rng.randint(0, 4)
        avg_q = (self.q1[state] + self.q2[state]) / 2
        return int(np.argmax(avg_q))

    def _apply_param_action(self, action):
        delta = 0.05
        if action == 0:
            self.pc = min(0.95, self.pc + delta)
        elif action == 1:
            self.pc = max(0.4, self.pc - delta)
        elif action == 2:
            self.pm = min(0.5, self.pm + delta)
        elif action == 3:
            self.pm = max(0.05, self.pm - delta)

    def _knowledge_guided_init(self):
        """4种知识引导初始化"""
        population = []
        n = self.pop_size
        # 策略1：关键路径导向
        for _ in range(n // 4):
            population.append(spt_chromosome(self.instance, self.rng))
        # 策略2：负载均衡导向
        for _ in range(n // 4):
            population.append(balance_chromosome(self.instance, self.rng))
        # 策略3：能耗导向
        for _ in range(n // 4):
            population.append(energy_chromosome(self.instance, self.rng))
        # 策略4：随机
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
            if self.rng.random() < self.pc:
                op_id = self.rng.randint(0, 5)
                _, op_func = CROSSOVER_OPERATORS[op_id]
                c1, c2 = op_func(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if self.rng.random() < self.pm:
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

    def _knowledge_guided_vns(self, chromosome: Chromosome) -> Chromosome:
        """知识引导VNS：根据当前解的瓶颈自动选择邻域"""
        best = chromosome
        best_obj = best.objectives.to_array()

        # 分析瓶颈
        workloads = np.zeros(self.instance.num_machines)
        for u in range(self.instance.num_machines):
            for (_, _, s, e) in best.schedule.machine_schedule.get(u, []):
                workloads[u] += e - s

        bottleneck_machine = int(np.argmax(workloads))

        for _ in range(3):
            candidate = best.copy()
            # 知识引导：重点调整瓶颈机器上的工序
            flat = 0
            for i in range(self.instance.num_jobs):
                for j in range(self.instance.num_operations[i]):
                    machines = self.instance.get_compatible_machines(i, j)
                    current_ma = candidate.ma[flat] % len(machines) if machines else 0
                    if machines and machines[current_ma] == bottleneck_machine:
                        # 将瓶颈机器上的工序转移到其他机器
                        alternatives = [idx for idx, m in enumerate(machines) if m != bottleneck_machine]
                        if alternatives and self.rng.random() < 0.3:
                            candidate.ma[flat] = self.rng.choice(alternatives)
                    flat += 1

            candidate.invalidate()
            cand_obj = candidate.objectives.to_array()
            if np.any(cand_obj < best_obj):
                best = candidate
                best_obj = cand_obj

        return best
