"""Integrity tests for the v5 adaptive-operator experiment runner."""

from types import SimpleNamespace

import numpy as np
import pytest

from data.loader import generate_random_instance
from experiments import run_revision_aos as runner


class _Objectives:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def to_array(self):
        return self._values.copy()


class _Solution:
    def __init__(self, name, values=(1.0, 1.0, 1.0)):
        self.name = name
        self.os = np.arange(6)
        self.objectives = _Objectives(values)
        self.invalidated = False

    def invalidate(self):
        self.invalidated = True


def _chromosome_fingerprint(chromosome):
    return tuple(
        np.asarray(values).tobytes()
        for values in (
            chromosome.os,
            chromosome.ma,
            chromosome.agv_assign,
            chromosome.agv_speed,
        )
    )


def _rng_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_archive_stably_deduplicates_objectives_before_capacity_truncation():
    first = _Solution("first", (1.0, 3.0, 3.0))
    duplicate = _Solution("duplicate", (1.0, 3.0, 3.0))
    middle = _Solution("middle", (2.0, 2.0, 2.0))
    last = _Solution("last", (3.0, 1.0, 1.0))

    archive = runner.update_archive(
        [first],
        [duplicate, middle, last],
        archive_size=3,
    )

    assert [item.name for item in archive] == ["first", "middle", "last"]
    assert len({tuple(item.objectives.to_array()) for item in archive}) == len(archive)

    metrics = runner.evaluate_archive([first, duplicate, middle, last])
    assert metrics["NSol"] == 3
    assert len(metrics["objectives"]) == 3


def test_generation_archive_keeps_nondominated_offspring_dropped_by_selection():
    parent = _Solution("parent", (2.0, 2.0, 2.0))
    dropped_offspring = _Solution("dropped", (1.0, 3.0, 2.0))
    selected_offspring = _Solution("selected", (3.0, 1.0, 2.0))

    archive = runner.update_generation_archive(
        [],
        [parent],
        [dropped_offspring, selected_offspring],
    )

    assert {item.name for item in archive} == {"parent", "dropped", "selected"}


def test_population_diversity_is_invariant_to_objective_scaling():
    objectives = np.array(
        [[1.0, 100.0, 0.5], [2.0, 80.0, 1.0], [4.0, 50.0, 2.0]]
    )
    scaled = objectives * np.array([1000.0, 0.001, 17.0])

    assert runner.population_diversity(scaled) == pytest.approx(
        runner.population_diversity(objectives)
    )


def test_initialization_uses_exact_30_20_20_30_quotas(monkeypatch):
    generated = {"spt": 0, "energy": 0, "balance": 0, "random": 0}

    def factory(kind):
        def generate(instance, rng):
            generated[kind] += 1
            return _Solution(f"{kind}-{generated[kind]}")

        return generate

    monkeypatch.setattr(runner, "spt_chromosome", factory("spt"))
    monkeypatch.setattr(runner, "energy_chromosome", factory("energy"))
    monkeypatch.setattr(runner, "balance_chromosome", factory("balance"))
    monkeypatch.setattr(runner, "random_chromosome", factory("random"))

    population = runner.initialize_population(
        SimpleNamespace(),
        100,
        np.random.RandomState(7),
    )

    assert len(population) == 100
    assert generated == {"spt": 30, "energy": 20, "balance": 20, "random": 30}
    assert runner.population_initialization_quotas(101) == (31, 20, 20, 30)


def test_methods_share_identical_initial_population_for_same_seed():
    instance = generate_random_instance(5, 4, 2, seed=17)
    fingerprints = []

    for variant in ("Random", "UniformFixed", "RandomUCBPPO", "AdaptiveSAOS"):
        init_rng, _, controller_rng = runner.make_rng_streams(43)
        # Constructing any selector must not consume the initialization stream.
        runner.make_selector(variant, controller_rng, "composite", 100, device="cpu")
        population = runner.initialize_population(instance, 20, init_rng)
        fingerprints.append(tuple(_chromosome_fingerprint(c) for c in population))

    assert all(fingerprint == fingerprints[0] for fingerprint in fingerprints[1:])


def test_random_transition_does_not_consume_evolution_rng():
    _, evolution_rng, controller_rng = runner.make_rng_streams(47)
    _, reference_evolution_rng, _ = runner.make_rng_streams(47)

    runner.make_selector(
        "RandomUCBPPO",
        controller_rng,
        "composite",
        100,
        device="cpu",
    )

    assert np.array_equal(
        evolution_rng.randint(0, 2**31 - 1, size=32),
        reference_evolution_rng.randint(0, 2**31 - 1, size=32),
    )


def test_uniform_fixed_offset_is_sampled_once_and_persists():
    rng = np.random.RandomState(53)
    selector = runner.UniformFixedSelector(rng, n_ops=7)

    first = selector.select(0, 20, 0, 0.0, 0.0)
    state_after_first = rng.get_state()
    sequence = [first] + [
        selector.select(gen, 20, 0, 0.0, 0.0)
        for gen in range(1, 15)
    ]

    assert sequence == [(gen + selector.offset) % 7 for gen in range(15)]
    assert _rng_states_equal(state_after_first, rng.get_state())
