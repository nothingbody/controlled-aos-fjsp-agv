"""Revision experiments for stage-aware adaptive operator selection.

This runner isolates the operator-selection module while keeping the same
FJSP-AGV encoding, operator library, decoding, and NSGA-III environmental
selection. It is intended for the major-revision experiments:

1. AOS direct baselines:
   Random, UniformFixed, ProbabilityMatching, AdaptivePursuit, UCB-only,
   PPO-only, fixed UCB->PPO, random UCB->PPO, adaptive SA-AOS.
2. Reward ablation under the adaptive SA-AOS controller.

The script writes one row per (dataset, instance, variant, seed), plus a
pickle containing each final non-dominated objective matrix. Unified HV can be
recomputed from those pickle files after the run.
"""

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.loader import load_benchmark_set
from src.algorithm.grl.ppo_agent import PPOAgent
from src.algorithm.nsga3.crossover import CROSSOVER_OPERATORS
from src.algorithm.nsga3.decoding import evaluate
from src.algorithm.nsga3.encoding import (
    balance_chromosome,
    energy_chromosome,
    random_chromosome,
    spt_chromosome,
)
from src.algorithm.nsga3.mutation import MUTATION_OPERATORS
from src.algorithm.nsga3.selection import (
    compute_hypervolume,
    generate_reference_points,
    non_dominated_sort,
    nsga3_select,
)
from src.utils.metrics import compute_spread


ALL_OPERATORS = {}
ALL_OPERATORS.update(CROSSOVER_OPERATORS)
ALL_OPERATORS.update(MUTATION_OPERATORS)
N_OPS = len(ALL_OPERATORS)
PROTOCOL_VERSION = "saos_bc_onpolicy_ppo_v5_20260720"

DATASETS = {
    "Brandimarte": "data/benchmarks/brandimarte",
    "Hurink_edata": "data/benchmarks/hurink_edata",
}

AOS_VARIANTS = [
    "Random",
    "UniformFixed",
    "ProbabilityMatching",
    "AdaptivePursuit",
    "UCBOnly",
    "PPOOnly",
    "FixedUCBPPO",
    "RandomUCBPPO",
    "AdaptiveNoBC",
    "AdaptiveSAOS",
]

REWARD_VARIANTS = {
    "R1_survival_only": "survival",
    "R2_hv_only": "hv",
    "R3_cmax_only": "cmax",
    "R4_survival_hv": "survival_hv",
    "R5_composite": "composite",
    "R6_adaptive_weight": "adaptive",
}


def _largest_remainder_quotas(total, weights):
    """Allocate an integer total deterministically according to ``weights``."""
    if total < 0:
        raise ValueError("total must be non-negative")
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError("weights must be a non-empty vector of non-negative values")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("weights must be finite and sum to a positive value")

    exact = total * weights / weights.sum()
    quotas = np.floor(exact).astype(int)
    remainder = int(total - quotas.sum())
    # Stable mergesort makes ties follow the declared component order.
    order = np.argsort(-(exact - quotas), kind="mergesort")
    quotas[order[:remainder]] += 1
    return quotas.tolist()


def population_initialization_quotas(pop_size):
    """Return exact SPT/energy/balance/random counts for the 30/20/20/30 mix."""
    return tuple(_largest_remainder_quotas(pop_size, (0.30, 0.20, 0.20, 0.30)))


def initialize_population(instance, pop_size, rng):
    """Create the declared 30/20/20/30 hybrid initial population exactly."""
    population = []
    n_spt, n_energy, n_balance, n_random = population_initialization_quotas(pop_size)

    # Retain both SPT seeds used by the original implementation: half are pure
    # SPT chromosomes and half receive a small OS perturbation. Odd counts place
    # the deterministic extra chromosome in the pure-SPT group.
    n_spt_pure = (n_spt + 1) // 2
    n_spt_perturbed = n_spt - n_spt_pure
    for _ in range(n_spt_pure):
        population.append(spt_chromosome(instance, rng))
    for _ in range(n_spt_perturbed):
        c = spt_chromosome(instance, rng)
        for _ in range(rng.randint(1, 5)):
            idx1, idx2 = rng.choice(len(c.os), 2, replace=False)
            c.os[idx1], c.os[idx2] = c.os[idx2], c.os[idx1]
        c.invalidate()
        population.append(c)
    for _ in range(n_energy):
        population.append(energy_chromosome(instance, rng))
    for _ in range(n_balance):
        population.append(balance_chromosome(instance, rng))
    for _ in range(n_random):
        population.append(random_chromosome(instance, rng))
    assert len(population) == pop_size
    for chrom in population:
        _ = chrom.objectives
    return population


def crowding_distance(objs):
    n, m = objs.shape
    cd = np.zeros(n)
    if n == 0:
        return cd
    for j in range(m):
        order = np.argsort(objs[:, j])
        cd[order[0]] = float("inf")
        cd[order[-1]] = float("inf")
        span = objs[order[-1], j] - objs[order[0], j]
        if span < 1e-10:
            continue
        for k in range(1, n - 1):
            cd[order[k]] += (objs[order[k + 1], j] - objs[order[k - 1], j]) / span
    return cd


def tournament_select(population, tournament_size, rng):
    indices = rng.choice(len(population), tournament_size, replace=False)
    candidates = [population[i] for i in indices]
    objs = np.array([c.objectives.to_array() for c in candidates])
    fronts = non_dominated_sort(objs)
    return candidates[fronts[0][0]]


def apply_operator(population, operator_id, pop_size, tournament_size, rng):
    offspring = []
    name, op_func = ALL_OPERATORS[operator_id]
    for _ in range(pop_size // 2):
        p1 = tournament_select(population, tournament_size, rng)
        p2 = tournament_select(population, tournament_size, rng)
        if operator_id < 5:
            c1, c2 = op_func(p1, p2, rng)
        else:
            if name in ("MachineReassign", "AGVReassign", "SpeedAdjust"):
                c1 = op_func(p1, mutation_rate=0.1, rng=rng)
                c2 = op_func(p2, mutation_rate=0.1, rng=rng)
            else:
                c1 = op_func(p1, rng=rng)
                c2 = op_func(p2, rng=rng)
        offspring.extend([c1, c2])
    offspring = offspring[:pop_size]
    for chrom in offspring:
        _ = chrom.objectives
    return offspring


def _objective_key(solution):
    return tuple(np.asarray(solution.objectives.to_array(), dtype=float).tolist())


def stable_unique_solutions(solutions):
    """Keep the first solution for each exact objective vector."""
    unique = []
    seen = set()
    for solution in solutions:
        key = _objective_key(solution)
        if key not in seen:
            seen.add(key)
            unique.append(solution)
    return unique


def stable_unique_objective_rows(objectives):
    """Return objective rows with exact duplicates removed in input order."""
    objectives = np.asarray(objectives, dtype=float)
    unique = []
    seen = set()
    for row in objectives:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if not unique:
        width = objectives.shape[1] if objectives.ndim == 2 else 0
        return np.empty((0, width), dtype=float)
    return np.asarray(unique, dtype=float)


def update_archive(archive, population, archive_size=100):
    # Deduplicate before sorting and truncation so repeated vectors cannot
    # consume archive capacity or bias crowding distances.
    all_solutions = stable_unique_solutions(archive + population)
    if not all_solutions:
        return []
    objs = np.array([c.objectives.to_array() for c in all_solutions])
    fronts = non_dominated_sort(objs)
    archive = [all_solutions[i] for i in fronts[0]]
    if len(archive) > archive_size:
        nd_objs = np.array([c.objectives.to_array() for c in archive])
        cd = crowding_distance(nd_objs)
        keep = np.argsort(-cd)[:archive_size]
        archive = [archive[i] for i in keep]
    return archive


def update_generation_archive(archive, parents, offspring, archive_size=100):
    """Update the archive from every solution evaluated in a generation.

    Environmental niching may discard a nondominated offspring from the next
    parent population. Updating from the complete evaluated parent-offspring
    pool ensures that such a solution is still considered by the bounded
    archive.
    """
    return update_archive(
        archive,
        list(parents) + list(offspring),
        archive_size=archive_size,
    )


def population_diversity(objectives):
    """Return the mean per-objective coefficient of variation."""
    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2 or len(objectives) == 0:
        raise ValueError("objectives must be a non-empty two-dimensional array")
    means = np.abs(np.mean(objectives, axis=0))
    scales = np.maximum(means, 1e-12)
    return float(np.mean(np.std(objectives, axis=0) / scales))


def normalized_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)) / math.log(len(counts)))


def compute_reward(scheme, survival, hv_delta, cmax_delta, gen, max_gen):
    if scheme == "survival":
        return float(survival)
    if scheme == "hv":
        return float(hv_delta)
    if scheme == "cmax":
        return float(cmax_delta)
    if scheme == "survival_hv":
        return float(0.6 * survival + 0.4 * hv_delta)
    if scheme == "adaptive":
        progress = gen / max(max_gen - 1, 1)
        alpha = 0.7 - 0.25 * progress
        beta = 0.2 + 0.20 * progress
        gamma = 1.0 - alpha - beta
        return float(alpha * survival + beta * hv_delta + gamma * cmax_delta)
    return float(0.5 * survival + 0.3 * hv_delta + 0.2 * cmax_delta)


class BaseSelector:
    def __init__(self, rng, n_ops=N_OPS, window=50, reward_scheme="composite"):
        self.rng = rng
        self.n_ops = n_ops
        self.window = window
        self.reward_scheme = reward_scheme
        self.history = []
        self.op_counts = np.zeros(n_ops, dtype=int)
        self.transition_gen = -1

    def build_state(self, gen, max_gen, stagnation, diversity, hv_trend, switched=False):
        recent = self.history[-self.window :]
        means = np.zeros(self.n_ops, dtype=np.float32)
        freq = np.zeros(self.n_ops, dtype=np.float32)
        total = max(len(recent), 1)
        for op_id, reward in recent:
            means[op_id] += reward
            freq[op_id] += 1
        for i in range(self.n_ops):
            if freq[i] > 0:
                means[i] /= freq[i]
            freq[i] /= total
        progress = np.array(
            [
                gen / max(max_gen, 1),
                stagnation / 20.0,
                diversity,
                hv_trend,
            ],
            dtype=np.float32,
        )
        return np.concatenate([means, freq, progress])

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        raise NotImplementedError

    def update(self, op_id, reward):
        self.history.append((op_id, reward))
        self.op_counts[op_id] += 1


class RandomSelector(BaseSelector):
    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        return int(self.rng.randint(0, self.n_ops))


class UniformFixedSelector(BaseSelector):
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)
        self.offset = None

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        if self.offset is None:
            self.offset = int(self.rng.randint(0, self.n_ops))
        return int((gen + self.offset) % self.n_ops)


class ProbabilityMatchingSelector(BaseSelector):
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)
        self.q = np.zeros(self.n_ops, dtype=float)
        self.n = np.zeros(self.n_ops, dtype=float)
        self.p_min = 0.02

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        scores = self.q - self.q.min() + 1e-6
        probs = self.p_min + (1 - self.n_ops * self.p_min) * scores / scores.sum()
        probs = probs / probs.sum()
        return int(self.rng.choice(self.n_ops, p=probs))

    def update(self, op_id, reward):
        super().update(op_id, reward)
        self.n[op_id] += 1
        self.q[op_id] += (reward - self.q[op_id]) / self.n[op_id]


class AdaptivePursuitSelector(ProbabilityMatchingSelector):
    def __init__(self, rng, **kwargs):
        super().__init__(rng, **kwargs)
        self.p = np.ones(self.n_ops, dtype=float) / self.n_ops
        self.p_min = 0.02
        self.p_max = 1 - (self.n_ops - 1) * self.p_min
        self.beta = 0.2

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        return int(self.rng.choice(self.n_ops, p=self.p / self.p.sum()))

    def update(self, op_id, reward):
        BaseSelector.update(self, op_id, reward)
        self.n[op_id] += 1
        self.q[op_id] += (reward - self.q[op_id]) / self.n[op_id]
        best = int(np.argmax(self.q))
        target = np.full(self.n_ops, self.p_min)
        target[best] = self.p_max
        self.p = self.p + self.beta * (target - self.p)
        self.p = self.p / self.p.sum()


class UCBSelector(BaseSelector):
    def __init__(self, rng, c=1.0, min_count=2, **kwargs):
        super().__init__(rng, **kwargs)
        self.c = c
        self.min_count = min_count

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        for i in range(self.n_ops):
            if self.op_counts[i] < self.min_count:
                return i
        recent = self.history[-self.window :]
        means = np.zeros(self.n_ops, dtype=float)
        counts = np.zeros(self.n_ops, dtype=float)
        for op_id, reward in recent:
            means[op_id] += reward
            counts[op_id] += 1
        total = max(counts.sum(), 1)
        ucb = np.zeros(self.n_ops, dtype=float)
        for i in range(self.n_ops):
            if counts[i] == 0:
                ucb[i] = float("inf")
            else:
                means[i] /= counts[i]
                ucb[i] = means[i] + self.c * np.sqrt(np.log(total) / counts[i])
        return int(np.argmax(ucb))


class PPOSelector(BaseSelector):
    def __init__(
        self,
        rng,
        device="cpu",
        lr=3e-4,
        rollout_size=16,
        state_dim=None,
        rng_seeds=None,
        **kwargs,
    ):
        super().__init__(rng, **kwargs)
        state_dim = self.n_ops * 2 + 4 if state_dim is None else int(state_dim)
        rng_seeds = dict(rng_seeds or {})
        self.ppo = PPOAgent(
            state_dim=state_dim,
            action_dim=self.n_ops,
            lr=lr,
            batch_size=rollout_size,
            device=device,
            network_seed=rng_seeds.get("network"),
            action_seed=rng_seeds.get("action"),
            bc_seed=rng_seeds.get("bc"),
            ppo_seed=rng_seeds.get("ppo"),
        )
        self.rollout_size = int(rollout_size)
        self.last_state = None
        self.last_action = None
        self.last_gen = -1
        self.last_max_gen = 0
        self.pending_ppo_update = False
        self.ppo_update_stats = []
        self.bc_stats = {}
        self.learning_time_seconds = 0.0
        self.bc_time_seconds = 0.0
        self.ppo_discarded_singletons = 0

    def _record_ppo_update(self, stats, context):
        if stats:
            record = dict(stats)
            record["update_context"] = str(context)
            self.ppo_update_stats.append(record)

    def _run_ppo_update(self, context, last_state=None):
        update_start = time.perf_counter()
        stats = self.ppo.update(last_state=last_state)
        update_seconds = time.perf_counter() - update_start
        self.learning_time_seconds += update_seconds
        if stats:
            stats = dict(stats)
            stats["update_seconds"] = float(update_seconds)
        self._record_ppo_update(stats, context)

    def _finish_ppo_step(self, reward):
        done = self.last_gen >= self.last_max_gen - 1
        self.ppo.store_reward(reward, done=done)
        if len(self.ppo.buffer) >= self.rollout_size:
            if done:
                self._run_ppo_update("terminal_full")
                self.pending_ppo_update = False
            else:
                self.pending_ppo_update = True

    def select(self, gen, max_gen, stagnation, diversity, hv_trend, switched=True):
        state = self.build_state(gen, max_gen, stagnation, diversity, hv_trend, switched)
        if self.pending_ppo_update:
            self._run_ppo_update("pre_action", last_state=state)
            self.pending_ppo_update = False
        action = self.ppo.select_and_store(state)
        self.last_state = state
        self.last_action = action
        self.last_gen = gen
        self.last_max_gen = max_gen
        return int(action)

    def update(self, op_id, reward):
        super().update(op_id, reward)
        self._finish_ppo_step(reward)

    def finalize(self):
        if len(self.ppo.buffer) >= 2:
            self._run_ppo_update("terminal_residual")
        elif len(self.ppo.buffer) == 1:
            self.ppo.buffer.clear()
            self.ppo_discarded_singletons += 1
        self.pending_ppo_update = False


class HybridSelector(PPOSelector):
    def __init__(
        self,
        rng,
        transition_mode="adaptive",
        fixed_transition=48,
        random_transition=None,
        min_per_op=3,
        min_buffer=30,
        ucb_c=1.0,
        use_behavior_cloning=True,
        min_stagnation=5,
        max_transition_fraction=0.70,
        **kwargs,
    ):
        super().__init__(rng, **kwargs)
        self.ucb = UCBSelector(
            rng,
            n_ops=self.n_ops,
            c=ucb_c,
            min_count=2,
            window=self.window,
            reward_scheme=self.reward_scheme,
        )
        self.transition_mode = transition_mode
        self.fixed_transition = fixed_transition
        self.random_transition = random_transition
        self.min_per_op = min_per_op
        self.min_buffer = min_buffer
        self.use_behavior_cloning = bool(use_behavior_cloning)
        self.min_stagnation = int(min_stagnation)
        self.max_transition_fraction = float(max_transition_fraction)
        self.switched = False
        self.demo_states = []
        self.demo_actions = []
        self.last_phase = "UCB"
        self.transition_reason = "none"

    def _should_switch(self, gen, max_gen, stagnation):
        if self.transition_mode == "fixed":
            ready = gen >= self.fixed_transition
            if ready:
                self.transition_reason = "fixed_generation"
            return ready
        if self.transition_mode == "random":
            ready = gen >= self.random_transition
            if ready:
                self.transition_reason = "random_generation"
            return ready
        coverage_ready = bool(
            np.all(self.ucb.op_counts >= self.min_per_op)
            and len(self.ucb.history) >= self.min_buffer
        )
        if not coverage_ready:
            return False
        if stagnation >= self.min_stagnation:
            self.transition_reason = "coverage_stagnation"
            return True
        latest_transition = int(math.ceil(self.max_transition_fraction * max_gen))
        if gen >= latest_transition:
            self.transition_reason = "coverage_latest_guard"
            return True
        return False

    def select(self, gen, max_gen, stagnation, diversity, hv_trend):
        if not self.switched and self._should_switch(gen, max_gen, stagnation):
            self.switched = True
            self.transition_gen = gen
            if self.use_behavior_cloning:
                bc_start = time.perf_counter()
                self.bc_stats = self.ppo.warm_start_behavior(
                    self.demo_states,
                    self.demo_actions,
                    n_epochs=100,
                    batch_size=32,
                )
                bc_seconds = time.perf_counter() - bc_start
                self.bc_time_seconds += bc_seconds
                self.learning_time_seconds += bc_seconds
        if self.switched:
            op_id = super().select(gen, max_gen, stagnation, diversity, hv_trend, True)
            self.last_phase = "PPO"
            return op_id
        state = self.build_state(gen, max_gen, stagnation, diversity, hv_trend, False)
        op_id = self.ucb.select(gen, max_gen, stagnation, diversity, hv_trend)
        self.demo_states.append(state.copy())
        self.demo_actions.append(int(op_id))
        self.last_state = state
        self.last_action = op_id
        self.last_gen = gen
        self.last_max_gen = max_gen
        self.last_phase = "UCB"
        return int(op_id)

    def update(self, op_id, reward):
        BaseSelector.update(self, op_id, reward)
        self.ucb.update(op_id, reward)
        if self.last_phase == "PPO":
            self._finish_ppo_step(reward)


def make_selector(variant, rng, reward_scheme, max_gen, device="cpu"):
    selector_variant = "AdaptiveSAOS" if variant in REWARD_VARIANTS else variant
    if selector_variant == "Random":
        return RandomSelector(rng, reward_scheme=reward_scheme)
    if selector_variant == "UniformFixed":
        return UniformFixedSelector(rng, reward_scheme=reward_scheme)
    if selector_variant == "ProbabilityMatching":
        return ProbabilityMatchingSelector(rng, reward_scheme=reward_scheme)
    if selector_variant == "AdaptivePursuit":
        return AdaptivePursuitSelector(rng, reward_scheme=reward_scheme)
    if selector_variant == "UCBOnly":
        return UCBSelector(rng, reward_scheme=reward_scheme)
    if selector_variant == "PPOOnly":
        return PPOSelector(rng, reward_scheme=reward_scheme, device=device)
    if selector_variant == "FixedUCBPPO":
        return HybridSelector(
            rng,
            reward_scheme=reward_scheme,
            transition_mode="fixed",
            fixed_transition=max(1, int(max_gen * 0.48)),
            device=device,
        )
    if selector_variant == "RandomUCBPPO":
        return HybridSelector(
            rng,
            reward_scheme=reward_scheme,
            transition_mode="random",
            random_transition=int(rng.randint(max(2, max_gen // 5), max(3, int(max_gen * 0.8)))),
            device=device,
        )
    if selector_variant == "AdaptiveNoBC":
        return HybridSelector(
            rng,
            reward_scheme=reward_scheme,
            transition_mode="adaptive",
            min_per_op=3,
            min_buffer=30,
            use_behavior_cloning=False,
            device=device,
        )
    if selector_variant == "AdaptiveSAOS":
        return HybridSelector(
            rng,
            reward_scheme=reward_scheme,
            transition_mode="adaptive",
            min_per_op=3,
            min_buffer=30,
            device=device,
        )
    raise ValueError(f"Unknown variant: {variant}")


def make_rng_streams(seed):
    """Create independent deterministic streams for init, evolution and AOS."""
    seed_sequence = np.random.SeedSequence(int(seed))
    child_sequences = seed_sequence.spawn(3)
    child_seeds = [int(seq.generate_state(1, dtype=np.uint32)[0]) for seq in child_sequences]
    return tuple(np.random.RandomState(child_seed) for child_seed in child_seeds)


def evaluate_archive(archive):
    if not archive:
        return {
            "HV_local": 0.0,
            "Spread": 0.0,
            "Cmax_best": float("inf"),
            "TEC_best": float("inf"),
            "WB_best": float("inf"),
            "NSol": 0,
            "objectives": np.empty((0, 3)),
        }
    objs = stable_unique_objective_rows(
        np.array([c.objectives.to_array() for c in archive])
    )
    fronts = non_dominated_sort(objs)
    nd = objs[fronts[0]]
    ref_point = nd.max(axis=0) * 1.2
    hv = compute_hypervolume(nd, ref_point)
    return {
        "HV_local": float(hv),
        "Spread": float(compute_spread(nd) if len(nd) > 2 else 0.0),
        "Cmax_best": float(nd[:, 0].min()),
        "TEC_best": float(nd[:, 1].min()),
        "WB_best": float(nd[:, 2].min()),
        "NSol": int(len(nd)),
        "objectives": nd,
    }


def run_single(task):
    dataset, inst_name, instance, variant, reward_scheme, seed, pop_size, max_gen, out_dir = task
    init_rng, evolution_rng, controller_rng = make_rng_streams(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    start = time.time()

    ref_points = generate_reference_points(3, 8)
    population = initialize_population(instance, pop_size, init_rng)
    selector = make_selector(
        variant,
        controller_rng,
        reward_scheme,
        max_gen,
        device="cpu",
    )
    archive = update_archive([], population)

    pop_objs = np.array([c.objectives.to_array() for c in population])
    ref_point = pop_objs.max(axis=0) * 1.1
    prev_hv = compute_hypervolume(pop_objs, ref_point)
    prev_best_cmax = pop_objs[:, 0].min()
    stagnation = 0
    hv_window = deque(maxlen=20)
    op_sequence = []
    reward_sequence = []
    hv_delta_sequence = []

    for gen in range(max_gen):
        pop_objs = np.array([c.objectives.to_array() for c in population])
        diversity = population_diversity(pop_objs)
        hv_trend = float(np.mean(hv_window)) if hv_window else 0.0

        op_id = selector.select(gen, max_gen, stagnation, diversity, hv_trend)
        offspring = apply_operator(population, op_id, pop_size, 5, evolution_rng)
        for chrom in offspring:
            chrom._revision_uid = id(chrom)

        combined = population + offspring
        combined_objs = np.array([c.objectives.to_array() for c in combined])
        archive = update_generation_archive(archive, population, offspring)
        selected = nsga3_select(
            combined_objs,
            pop_size,
            ref_points,
            rng=evolution_rng,
        )
        new_population = [combined[i] for i in selected]

        selected_ids = {getattr(c, "_revision_uid", None) for c in new_population}
        survived = sum(1 for c in offspring if getattr(c, "_revision_uid", None) in selected_ids)
        survival = survived / max(len(offspring), 1)

        new_objs = np.array([c.objectives.to_array() for c in new_population])
        current_hv = compute_hypervolume(new_objs, ref_point)
        hv_delta = np.clip((current_hv - prev_hv) / max(abs(prev_hv), 1e-10), -1.0, 1.0)
        current_best_cmax = new_objs[:, 0].min()
        cmax_delta = np.clip(
            (prev_best_cmax - current_best_cmax) / max(abs(prev_best_cmax), 1e-10),
            -1.0,
            1.0,
        )
        reward = compute_reward(reward_scheme, survival, hv_delta, cmax_delta, gen, max_gen)
        selector.update(op_id, reward)

        hv_window.append(float(hv_delta))
        op_sequence.append(int(op_id))
        reward_sequence.append(float(reward))
        hv_delta_sequence.append(float(hv_delta))
        stagnation = stagnation + 1 if current_hv <= prev_hv * 1.001 else 0
        prev_hv = current_hv
        prev_best_cmax = current_best_cmax
        population = new_population

    if hasattr(selector, "finalize"):
        selector.finalize()

    metrics = evaluate_archive(archive)
    counts = np.bincount(np.array(op_sequence, dtype=int), minlength=N_OPS)
    last_counts = np.bincount(np.array(op_sequence[-20:], dtype=int), minlength=N_OPS)
    if len(reward_sequence) > 2 and np.std(reward_sequence) > 1e-12 and np.std(hv_delta_sequence) > 1e-12:
        reward_hv_corr = float(np.corrcoef(reward_sequence, hv_delta_sequence)[0, 1])
    else:
        reward_hv_corr = 0.0

    ppo_stats = getattr(selector, "ppo_update_stats", [])
    bc_stats = getattr(selector, "bc_stats", {})

    def mean_stat(key):
        values = [float(item[key]) for item in ppo_stats if key in item]
        return float(np.mean(values)) if values else 0.0

    pkl_dir = Path(out_dir) / "fronts"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = pkl_dir / f"{dataset}_{Path(inst_name).stem}_{variant}_{reward_scheme}_seed{seed}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "dataset": dataset,
                "instance": inst_name,
                "variant": variant,
                "reward_scheme": reward_scheme,
                "seed": seed,
                "objectives": metrics.pop("objectives"),
                "operator_sequence": op_sequence,
                "reward_sequence": reward_sequence,
                "hv_delta_sequence": hv_delta_sequence,
                "ppo_update_stats": ppo_stats,
                "behavior_cloning_stats": bc_stats,
            },
            f,
        )

    row = {
        "Protocol": PROTOCOL_VERSION,
        "dataset": dataset,
        "instance": inst_name,
        "variant": variant,
        "reward_scheme": reward_scheme,
        "seed": seed,
        **metrics,
        "Time": float(time.time() - start),
        "Execution_device": "cpu",
        "Extension_seed_scheme": "sha256(dataset/filename)+42",
        "Torch_CUDA_available": bool(torch.cuda.is_available()),
        "Entropy_all": normalized_entropy(counts),
        "Entropy_last20": normalized_entropy(last_counts),
        "Transition_gen": int(selector.transition_gen),
        "Transition_reason": str(getattr(selector, "transition_reason", "none")),
        "Reward_HV_corr": reward_hv_corr,
        "PPO_update_count": int(len(ppo_stats)),
        "PPO_rollout_sizes": json.dumps(
            [int(item.get("rollout_size", 0)) for item in ppo_stats],
            separators=(",", ":"),
        ),
        "PPO_policy_loss_mean": mean_stat("policy_loss"),
        "PPO_value_loss_mean": mean_stat("value_loss"),
        "PPO_entropy_mean": mean_stat("entropy"),
        "BC_samples": int(bc_stats.get("bc_samples", 0)),
        "BC_loss": float(bc_stats.get("bc_loss", 0.0)),
        "BC_accuracy": float(bc_stats.get("bc_accuracy", 0.0)),
        "BC_entropy": float(bc_stats.get("bc_entropy", 0.0)),
        "BC_final_loss": float(bc_stats.get("bc_final_loss", 0.0)),
        "BC_final_accuracy": float(bc_stats.get("bc_final_accuracy", 0.0)),
        "BC_final_entropy": float(bc_stats.get("bc_final_entropy", 0.0)),
        "Operator_counts": json.dumps(counts.tolist(), separators=(",", ":")),
        "front_pickle": str(pkl_path),
    }
    return row


def load_instances(limit=None):
    loaded = []
    for dataset, path in DATASETS.items():
        for inst_name, instance in load_benchmark_set(path, num_agv=3):
            loaded.append((dataset, inst_name, instance))
    if limit:
        return loaded[:limit]
    return loaded


def existing_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(
                (
                    row["dataset"],
                    row["instance"],
                    row["variant"],
                    row["reward_scheme"],
                    int(row["seed"]),
                )
            )
    return keys


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks(args):
    instances = load_instances(limit=args.limit_instances)
    seeds = list(range(args.seed_start, args.seed_end))
    tasks = []
    if args.experiment == "aos":
        variants = args.variants.split(",") if args.variants else AOS_VARIANTS
        for dataset, inst_name, instance in instances:
            for variant in variants:
                for seed in seeds:
                    tasks.append(
                        (
                            dataset,
                            inst_name,
                            instance,
                            variant,
                            args.reward_scheme,
                            seed,
                            args.pop_size,
                            args.max_gen,
                            args.out_dir,
                        )
                    )
    elif args.experiment == "reward":
        reward_items = [
            item
            for item in REWARD_VARIANTS.items()
            if not args.variants or item[0] in args.variants.split(",")
        ]
        for dataset, inst_name, instance in instances:
            for label, scheme in reward_items:
                for seed in seeds:
                    tasks.append(
                        (
                            dataset,
                            inst_name,
                            instance,
                            label,
                            scheme,
                            seed,
                            args.pop_size,
                            args.max_gen,
                            args.out_dir,
                        )
                    )
    else:
        raise ValueError(args.experiment)
    done = existing_keys(args.result_file)
    return [t for t in tasks if (t[0], t[1], t[3], t[4], t[5]) not in done]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["aos", "reward"], required=True)
    parser.add_argument("--out-dir", default="results/revision")
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=52)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--limit-instances", type=int, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--reward-scheme", default="composite")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.result_file is None:
        args.result_file = os.path.join(args.out_dir, f"{args.experiment}_runs.csv")

    tasks = build_tasks(args)
    total = len(tasks)
    print(f"[revision] experiment={args.experiment} tasks={total} result={args.result_file}", flush=True)
    if total == 0:
        print("[revision] nothing to do", flush=True)
        return

    start = time.time()
    completed = 0
    n_workers = max(1, min(args.workers, total, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_map = {executor.submit(run_single, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            completed += 1
            try:
                row = future.result()
                append_row(args.result_file, row)
                msg = (
                    f"[{completed}/{total}] {row['dataset']}/{row['instance']} "
                    f"{row['variant']} {row['reward_scheme']} seed={row['seed']} "
                    f"HV={row['HV_local']:.3g} Cmax={row['Cmax_best']:.1f} "
                    f"NSol={row['NSol']} T={row['Time']:.1f}s"
                )
            except Exception as exc:
                dataset, inst_name, _, variant, reward_scheme, seed, *_ = task
                msg = (
                    f"[{completed}/{total}] ERROR {dataset}/{inst_name} "
                    f"{variant} {reward_scheme} seed={seed}: {exc!r}"
                )
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / max(rate, 1e-9)
            print(f"{msg} ETA={eta/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
