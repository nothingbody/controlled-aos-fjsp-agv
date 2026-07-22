"""AOS-CPEA 主框架：双层自适应算子选择 + 关键路径RL改进

核心设计：
  Module A（双层AOS）:
    Phase 1 (gen < T_switch): UCB 快速探索，同时收集 (state, action, reward) 给 PPO
    Phase 2 (gen >= T_switch): PPO 接管，利用进化状态特征做智能决策
    T_switch 自适应确定：当每个算子都被用过 >=3 次且 buffer >= min_buffer

  Module B（关键路径RL改进）:
    PPO-guided 关键路径定向改进，每 τ 代激活一次

  复合奖励：α·survival_rate + β·ΔHV_norm + γ·ΔCmax_norm
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm

from src.problem.instance import FJSPAGVInstance
from src.problem.graph_builder import build_hetero_graph
from src.algorithm.nsga3.encoding import (
    Chromosome, random_chromosome, spt_chromosome,
    energy_chromosome, balance_chromosome
)
from src.algorithm.nsga3.decoding import decode, evaluate, Objectives
from src.algorithm.nsga3.crossover import CROSSOVER_OPERATORS
from src.algorithm.nsga3.mutation import MUTATION_OPERATORS
from src.algorithm.nsga3.parallel_eval import parallel_evaluate
from src.algorithm.nsga3.selection import (
    generate_reference_points, nsga3_select,
    non_dominated_sort, compute_hypervolume
)
from src.algorithm.grl.ppo_agent import PPOAgent

ALL_OPERATORS = {}
ALL_OPERATORS.update(CROSSOVER_OPERATORS)
ALL_OPERATORS.update(MUTATION_OPERATORS)


class SlidingWindowUCB:
    """滑动窗口UCB1——双层AOS的Phase 1组件"""

    def __init__(self, n_operators, window_size=50, c=1.0):
        self.n_ops = n_operators
        self.window_size = window_size
        self.c = c
        self.history = []
        self.op_counts = np.zeros(n_operators, dtype=int)

    def select(self, rng=None):
        for i in range(self.n_ops):
            if self.op_counts[i] < 2:
                return i
        recent = self.history[-self.window_size:]
        mean_rewards = np.zeros(self.n_ops)
        counts = np.zeros(self.n_ops)
        for op_id, reward in recent:
            mean_rewards[op_id] += reward
            counts[op_id] += 1
        total = max(sum(counts), 1)
        ucb = np.zeros(self.n_ops)
        for i in range(self.n_ops):
            if counts[i] == 0:
                ucb[i] = float('inf')
            else:
                mean_rewards[i] /= counts[i]
                ucb[i] = mean_rewards[i] + self.c * np.sqrt(np.log(total) / counts[i])
        return int(np.argmax(ucb))

    def update(self, operator_id, reward):
        self.history.append((operator_id, reward))
        self.op_counts[operator_id] += 1

    def is_warmed_up(self, min_per_op=3):
        return all(self.op_counts[i] >= min_per_op for i in range(self.n_ops))


class DualLayerAOS:
    """双层自适应算子选择：UCB探索 → PPO决策

    Phase 1 (UCB): 零预热，快速积累经验
    Phase 2 (PPO): 利用进化状态特征做智能长期决策
    过渡条件：每个算子至少用3次 且 buffer >= min_switch_buffer
    """

    def __init__(self, n_operators, device='cpu',
                 ucb_window=50, ucb_c=1.0,
                 min_switch_buffer=48,
                 reward_alpha=0.5, reward_beta=0.3, reward_gamma=0.2):
        self.n_ops = n_operators
        self.device = device

        # Phase 1: UCB
        self.ucb = SlidingWindowUCB(n_operators, ucb_window, ucb_c)

        # Phase 2: PPO
        # 状态维度: 10(算子均值奖励) + 10(算子使用频率) + 5(进化进度特征) = 25
        self.state_dim = n_operators * 2 + 4
        self.ppo = PPOAgent(
            state_dim=self.state_dim,
            action_dim=n_operators,
            lr=3e-4, device=device
        )

        # 过渡控制
        self.min_switch_buffer = min_switch_buffer
        self.switched = False
        self.switch_gen = -1

        # 复合奖励权重
        self.alpha = reward_alpha  # survival_rate
        self.beta = reward_beta    # ΔHV
        self.gamma = reward_gamma  # ΔCmax

        # 历史记录
        self.phase_history = []  # [(gen, phase, operator_id)]

    def _build_state(self, gen, max_gen, stagnation, pop_diversity, hv_trend):
        """构建PPO状态向量 ∈ R^25"""
        # 算子统计（从UCB history提取）
        recent = self.ucb.history[-self.ucb.window_size:]
        mean_rewards = np.zeros(self.n_ops, dtype=np.float32)
        freq = np.zeros(self.n_ops, dtype=np.float32)
        total = max(len(recent), 1)
        for op_id, reward in recent:
            mean_rewards[op_id] += reward
            freq[op_id] += 1
        for i in range(self.n_ops):
            if freq[i] > 0:
                mean_rewards[i] /= freq[i]
            freq[i] /= total

        # 进化进度特征
        progress = np.array([
            gen / max(max_gen, 1),
            stagnation / 20.0,
            pop_diversity,
            hv_trend,
        ], dtype=np.float32)

        return np.concatenate([mean_rewards, freq, progress])

    def select(self, gen, max_gen, stagnation, pop_diversity, hv_trend, rng=None):
        """选择算子——自动判断使用UCB还是PPO"""
        # 检查是否满足过渡条件
        if not self.switched:
            if (self.ucb.is_warmed_up(min_per_op=3) and
                    len(self.ucb.history) >= self.min_switch_buffer):
                self.switched = True
                self.switch_gen = gen
                # 用UCB积累的数据做一次PPO更新
                if len(self.ppo.buffer) >= 32:
                    self.ppo.update()

        if not self.switched:
            # Phase 1: UCB
            op_id = self.ucb.select(rng)
            # 同时存储经验给PPO（为过渡做准备）
            state = self._build_state(gen, max_gen, stagnation, pop_diversity, hv_trend)
            self.ppo.select_and_store(state)  # 存储state+action(PPO自己选的)
            # 但实际用UCB选的算子——PPO的action会在update时被覆盖
            self._last_state = state
            self._last_phase = 'UCB'
        else:
            # Phase 2: PPO
            state = self._build_state(gen, max_gen, stagnation, pop_diversity, hv_trend)
            op_id = self.ppo.select_and_store(state)
            self._last_state = state
            self._last_phase = 'PPO'

        self.phase_history.append((gen, self._last_phase, op_id))
        return op_id

    def compute_reward(self, survival_rate, hv_improvement, cmax_improvement):
        """计算复合奖励"""
        return (self.alpha * survival_rate +
                self.beta * hv_improvement +
                self.gamma * cmax_improvement)

    def update(self, operator_id, reward):
        """更新UCB和PPO"""
        # 始终更新UCB（保持统计）
        self.ucb.update(operator_id, reward)
        # 更新PPO buffer中最后一条的reward
        self.ppo.store_reward(reward)

    def try_ppo_update(self, min_buffer=64):
        """尝试PPO参数更新"""
        if self.switched and len(self.ppo.buffer) >= min_buffer:
            self.ppo.update()

    @property
    def current_phase(self):
        return 'PPO' if self.switched else 'UCB'


class GRLEA:
    """AOS-CPEA: 双层RL-AOS + 关键路径RL改进"""

    def __init__(self, instance: FJSPAGVInstance,
                 pop_size: int = 200,
                 max_gen: int = 500,
                 tournament_size: int = 5,
                 ref_divisions: int = 8,
                 hidden_dim: int = 64,
                 use_grl: bool = True,
                 top_k_improve: int = 20,
                 improve_steps: int = 5,
                 improve_interval: int = 3,
                 device: str = 'auto',
                 seed: int = 42,
                 verbose: bool = True):

        self.instance = instance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.tournament_size = tournament_size
        self.use_grl = use_grl
        self.top_k_improve = top_k_improve
        self.improve_steps = improve_steps
        self.improve_interval = improve_interval
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.rng = np.random.RandomState(seed)
        self.verbose = verbose
        self.hidden_dim = hidden_dim
        self.num_objectives = 3
        self.ref_points = generate_reference_points(self.num_objectives, ref_divisions)

        # Module A: 双层AOS
        self.aos = DualLayerAOS(
            n_operators=len(ALL_OPERATORS),
            device=self.device,
            ucb_window=50, ucb_c=1.0,
            min_switch_buffer=48,
            reward_alpha=0.5, reward_beta=0.3, reward_gamma=0.2
        )

        # Module B: 关键路径RL改进
        state_dim_b = hidden_dim + 4
        self.ppo_b = PPOAgent(
            state_dim=state_dim_b, action_dim=20,
            lr=3e-4, device=self.device
        )

        self.hat_encoder = None
        self.archive = []
        self.archive_size = 100
        self.history = {'hv': [], 'gen': [], 'phase': []}

    def _init_hat(self):
        if self.hat_encoder is None:
            try:
                from src.algorithm.grl.het_gat import HATEncoder
                self.hat_encoder = HATEncoder(
                    hidden_dim=self.hidden_dim,
                    num_heads=4, num_layers=3, dropout=0.1
                ).to(self.device)
                self.hat_encoder.eval()
            except ImportError:
                self.hat_encoder = None

    def _get_graph_embedding(self, chromosome: Chromosome) -> np.ndarray:
        if self.hat_encoder is not None:
            try:
                graph = build_hetero_graph(self.instance, chromosome.schedule)
                graph.to(self.device)
                with torch.no_grad():
                    emb = self.hat_encoder(graph)
                return emb.cpu().numpy()
            except Exception:
                if not hasattr(self, '_hat_warn_printed'):
                    self._hat_warn_printed = True
        obj = chromosome.objectives
        features = np.array([
            obj.makespan / 10000.0,
            obj.total_energy / 100000.0,
            obj.workload_balance / 100.0,
        ], dtype=np.float32)
        emb = np.tile(features, self.hidden_dim // 3 + 1)[:self.hidden_dim]
        return emb.astype(np.float32)

    def _get_solution_state_b(self, chromosome: Chromosome) -> np.ndarray:
        from src.algorithm.grl.critical_path import compute_bottleneck_features
        g_sol = self._get_graph_embedding(chromosome)
        f_bottleneck = compute_bottleneck_features(
            chromosome.schedule, self.instance, chromosome.objectives.makespan
        )
        return np.concatenate([g_sol, f_bottleneck])

    def _crowding_distance(self, objs: np.ndarray) -> np.ndarray:
        n, m = objs.shape
        cd = np.zeros(n)
        for j in range(m):
            sorted_idx = np.argsort(objs[:, j])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            obj_range = objs[sorted_idx[-1], j] - objs[sorted_idx[0], j]
            if obj_range < 1e-10:
                continue
            for k in range(1, n - 1):
                cd[sorted_idx[k]] += (
                    objs[sorted_idx[k + 1], j] - objs[sorted_idx[k - 1], j]
                ) / obj_range
        return cd

    def initialize_population(self) -> list:
        population = []
        n = self.pop_size
        n1 = int(n * 0.3)
        for _ in range(n1 // 3 + 1):
            population.append(spt_chromosome(self.instance, self.rng))
        for _ in range(n1 // 3 + 1):
            c = spt_chromosome(self.instance, self.rng)
            swap_count = self.rng.randint(1, 5)
            for _ in range(swap_count):
                idx1, idx2 = self.rng.choice(len(c.os), 2, replace=False)
                c.os[idx1], c.os[idx2] = c.os[idx2], c.os[idx1]
                c.invalidate()
            population.append(c)
        n2 = int(n * 0.2)
        for _ in range(n2):
            population.append(energy_chromosome(self.instance, self.rng))
        n3 = int(n * 0.2)
        for _ in range(n3):
            population.append(balance_chromosome(self.instance, self.rng))
        while len(population) < n:
            population.append(random_chromosome(self.instance, self.rng))
        population = population[:n]
        for c in population:
            _ = c.objectives
        return population

    def tournament_select(self, population: list) -> Chromosome:
        indices = self.rng.choice(len(population), self.tournament_size, replace=False)
        candidates = [population[i] for i in indices]
        objs = np.array([c.objectives.to_array() for c in candidates])
        fronts = non_dominated_sort(objs)
        return candidates[fronts[0][0]]

    def apply_operator(self, population: list, operator_id: int) -> list:
        offspring = []
        name, op_func = ALL_OPERATORS[operator_id]
        for _ in range(self.pop_size // 2):
            p1 = self.tournament_select(population)
            p2 = self.tournament_select(population)
            if operator_id < 5:
                c1, c2 = op_func(p1, p2, self.rng)
                offspring.extend([c1, c2])
            else:
                if name in ('MachineReassign', 'AGVReassign', 'SpeedAdjust'):
                    c1 = op_func(p1, mutation_rate=0.1, rng=self.rng)
                    c2 = op_func(p2, mutation_rate=0.1, rng=self.rng)
                else:
                    c1 = op_func(p1, rng=self.rng)
                    c2 = op_func(p2, rng=self.rng)
                offspring.extend([c1, c2])
        for c in offspring[:self.pop_size]:
            _ = c.objectives
        return offspring[:self.pop_size]

    def local_improve(self, offspring: list) -> list:
        if not self.use_grl or len(offspring) < self.top_k_improve:
            return []
        objs = np.array([c.objectives.to_array() for c in offspring])
        fronts = non_dominated_sort(objs)
        front0 = fronts[0]
        if len(front0) >= self.top_k_improve:
            cd = self._crowding_distance(objs[front0])
            top_in_front = np.argsort(-cd)[:self.top_k_improve]
            top_indices = [front0[i] for i in top_in_front]
        else:
            top_indices = list(front0)
            if len(fronts) > 1:
                remaining = self.top_k_improve - len(top_indices)
                top_indices.extend(fronts[1][:remaining])

        improved = []
        for idx in top_indices:
            sol = offspring[idx].copy()
            for step in range(self.improve_steps):
                state_b = self._get_solution_state_b(sol)
                action = self.ppo_b.select(state_b)
                improved_sol = self._apply_improvement(sol, action)
                if improved_sol is not None:
                    old_obj = sol.objectives.to_array()
                    new_obj = improved_sol.objectives.to_array()
                    if np.any(new_obj < old_obj):
                        reward = float(np.mean((old_obj - new_obj) / (old_obj + 1e-6)))
                        sol = improved_sol
                    else:
                        reward = -0.01
                else:
                    reward = -0.05
                self.ppo_b.store_reward(reward)
            improved.append(sol)
        return improved

    def _apply_improvement(self, chromosome: Chromosome, action: int) -> Optional[Chromosome]:
        from src.algorithm.grl.critical_path import find_critical_path
        child = chromosome.copy()
        inst = child.instance
        critical_ops = find_critical_path(child.schedule, inst)
        if not critical_ops:
            flat_idx = self.rng.randint(0, child.n_total)
            i, j = self._flat_to_op(inst, flat_idx)
            critical_ops = [(i, j)] if i is not None else []
        if not critical_ops:
            return None
        max_targets = min(len(critical_ops), 10)
        target_idx = action % max_targets
        i, j = critical_ops[target_idx]
        flat_idx = self._op_to_flat(inst, i, j)
        if flat_idx is None or flat_idx >= child.n_total:
            return None
        action_type = (action // max_targets) % 4
        if action_type == 0:
            machines = inst.get_compatible_machines(i, j)
            if len(machines) > 1:
                current_ma = child.ma[flat_idx] % len(machines)
                candidates = [m for m_idx, m in enumerate(machines) if m_idx != current_ma]
                if candidates:
                    times = [inst.get_processing_time(i, j, m) for m in candidates]
                    best = candidates[int(np.argmin(times))]
                    child.ma[flat_idx] = machines.index(best)
        elif action_type == 1:
            agv_loads = np.zeros(inst.num_agv)
            for idx2 in range(child.n_total):
                agv_loads[child.agv_assign[idx2] % inst.num_agv] += 1
            child.agv_assign[flat_idx] = int(np.argmin(agv_loads))
        elif action_type == 2:
            child.agv_speed[flat_idx] = (child.agv_speed[flat_idx] + 1) % inst.num_speeds
        elif action_type == 3:
            if target_idx + 1 < len(critical_ops):
                next_op = critical_ops[target_idx + 1]
                if next_op[0] != i:
                    pos1 = self._find_op_in_os(child.os, i, j, inst)
                    pos2 = self._find_op_in_os(child.os, next_op[0], next_op[1], inst)
                    if pos1 is not None and pos2 is not None:
                        child.os[pos1], child.os[pos2] = child.os[pos2], child.os[pos1]
        child.invalidate()
        return child

    def _op_to_flat(self, inst, job, op):
        idx = 0
        for i in range(inst.num_jobs):
            for j in range(inst.num_operations[i]):
                if i == job and j == op:
                    return idx
                idx += 1
        return None

    def _find_op_in_os(self, os_seq, job, op, inst):
        count = 0
        for pos in range(len(os_seq)):
            if os_seq[pos] == job:
                if count == op:
                    return pos
                count += 1
        return None

    def _flat_to_op(self, inst, flat_idx):
        idx = 0
        for i in range(inst.num_jobs):
            for j in range(inst.num_operations[i]):
                if idx == flat_idx:
                    return i, j
                idx += 1
        return None, None

    def update_archive(self, population: list):
        all_solutions = self.archive + population
        objs = np.array([c.objectives.to_array() for c in all_solutions])
        fronts = non_dominated_sort(objs)
        self.archive = [all_solutions[i] for i in fronts[0]]
        if len(self.archive) > self.archive_size:
            objs_nd = np.array([c.objectives.to_array() for c in self.archive])
            cd = self._crowding_distance(objs_nd)
            keep = np.argsort(-cd)[:self.archive_size]
            self.archive = [self.archive[i] for i in keep]

    def run(self) -> list:
        import gc
        if self.use_grl:
            self._init_hat()

        if self.verbose:
            print(f"Initializing population (size={self.pop_size})...")
        population = self.initialize_population()
        self.update_archive(population)

        objs = np.array([c.objectives.to_array() for c in population])
        ref_point = objs.max(axis=0) * 1.1
        prev_hv = compute_hypervolume(objs, ref_point)
        prev_best_cmax = objs[:, 0].min()
        stagnation = 0
        hv_window = []

        for c in population:
            c._id = id(c)

        iterator = range(self.max_gen)
        if self.verbose:
            iterator = tqdm(iterator, desc="AOS-CPEA")

        for gen in iterator:
            # === Module A: 双层AOS算子选择 ===
            if self.use_grl:
                # 计算种群多样性
                pop_objs = np.array([c.objectives.to_array() for c in population])
                diversity = float(np.std(pop_objs, axis=0).mean() / (np.mean(pop_objs) + 1e-10))
                hv_trend = float(np.mean(hv_window[-5:])) if len(hv_window) >= 5 else 0.0

                operator_id = self.aos.select(
                    gen, self.max_gen, stagnation, diversity, hv_trend, self.rng
                )
            else:
                operator_id = self.rng.randint(0, len(ALL_OPERATORS))

            # 生成子代
            offspring = self.apply_operator(population, operator_id)
            for c in offspring:
                c._id = id(c)

            # === Module B: 关键路径RL改进 ===
            improved = []
            if self.use_grl and gen % self.improve_interval == 0:
                improved = self.local_improve(offspring)
                for c in improved:
                    c._id = id(c)

            # === NSGA-III环境选择 ===
            combined = population + offspring + improved
            combined_objs = np.array([c.objectives.to_array() for c in combined])
            selected_indices = nsga3_select(combined_objs, self.pop_size, self.ref_points)
            new_population = [combined[i] for i in selected_indices]

            # === 计算复合奖励 ===
            if self.use_grl:
                # 1) 存活率
                selected_ids = set(id(c) for c in new_population)
                n_survived = sum(1 for c in offspring if c._id in selected_ids)
                survival_rate = n_survived / max(len(offspring), 1)

                # 2) HV改善
                new_pop_objs = np.array([c.objectives.to_array() for c in new_population])
                current_hv = compute_hypervolume(new_pop_objs, ref_point)
                hv_improvement = (current_hv - prev_hv) / max(abs(prev_hv), 1e-10)
                hv_improvement = np.clip(hv_improvement, -1.0, 1.0)

                # 3) Cmax改善
                current_best_cmax = new_pop_objs[:, 0].min()
                cmax_improvement = (prev_best_cmax - current_best_cmax) / max(abs(prev_best_cmax), 1e-10)
                cmax_improvement = np.clip(cmax_improvement, -1.0, 1.0)

                # 复合奖励
                reward = self.aos.compute_reward(survival_rate, hv_improvement, cmax_improvement)
                self.aos.update(operator_id, reward)
                self.aos.try_ppo_update(min_buffer=64)

                # HV趋势追踪
                hv_window.append(hv_improvement)
                if len(hv_window) > 20:
                    hv_window.pop(0)

                prev_best_cmax = current_best_cmax
            else:
                new_pop_objs = np.array([c.objectives.to_array() for c in new_population])
                current_hv = compute_hypervolume(new_pop_objs, ref_point)

            population = new_population
            self.update_archive(population)

            # Module B PPO更新
            if self.use_grl and len(self.ppo_b.buffer) >= 64:
                self.ppo_b.update()

            # 停滞检测
            if current_hv <= prev_hv * 1.001:
                stagnation += 1
            else:
                stagnation = 0
            prev_hv = current_hv

            self.history['hv'].append(current_hv)
            self.history['gen'].append(gen)
            self.history['phase'].append(self.aos.current_phase if self.use_grl else 'N/A')

            if gen % 50 == 0:
                gc.collect()

            if self.verbose and isinstance(iterator, tqdm):
                best_makespan = min(c.objectives.makespan for c in population)
                best_energy = min(c.objectives.total_energy for c in population)
                iterator.set_postfix({
                    'HV': f'{current_hv:.2f}',
                    'Cmax': f'{best_makespan:.1f}',
                    'TEC': f'{best_energy:.1f}',
                    'Op': ALL_OPERATORS[operator_id][0],
                    'Ph': self.aos.current_phase if self.use_grl else '-',
                    'Arc': len(self.archive),
                })

        if self.verbose:
            print(f"\nOptimization complete. Archive size: {len(self.archive)}")
            print(f"Best makespan: {min(c.objectives.makespan for c in self.archive):.2f}")
            print(f"Best energy: {min(c.objectives.total_energy for c in self.archive):.2f}")
            if self.use_grl and self.aos.switch_gen >= 0:
                print(f"AOS phase transition: UCB→PPO at gen {self.aos.switch_gen}")

        return self.archive
