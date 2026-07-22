"""Pytest fixtures for the legacy executable-style test modules."""

import numpy as np
import pytest

from data.loader import generate_random_instance
from src.algorithm.nsga3.encoding import (
    balance_chromosome,
    energy_chromosome,
    random_chromosome,
    spt_chromosome,
)
from src.environment.dynamic_events import EventGenerator


@pytest.fixture
def inst():
    return generate_random_instance(10, 6, 3, seed=42)


@pytest.fixture
def chroms(inst):
    rng = np.random.RandomState(42)
    return {
        "random": random_chromosome(inst, rng),
        "spt": spt_chromosome(inst, rng),
        "energy": energy_chromosome(inst, rng),
        "balance": balance_chromosome(inst, rng),
    }


@pytest.fixture
def events(inst):
    generator = EventGenerator(
        inst,
        new_job_rate=0.3,
        machine_breakdown_rate=0.03,
        agv_breakdown_rate=0.02,
        seed=42,
    )
    return generator.generate_events(time_horizon=500, start_time=50)


@pytest.fixture
def chrom(inst):
    return random_chromosome(inst, np.random.RandomState(42))
