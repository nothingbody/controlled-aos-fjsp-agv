from types import SimpleNamespace

import numpy as np
import pytest

from src.algorithm.nsga3.encoding import energy_chromosome
from src.problem.energy_model import compute_machine_energy
from src.problem.instance import FJSPAGVInstance


def _instance(num_machines=2):
    return FJSPAGVInstance(
        num_jobs=2,
        num_machines=num_machines,
        num_agv=1,
        num_operations=[1, 1],
        processing_times={(0, 0, 0): 10.0, (1, 0, 1): 20.0},
        compatible_machines={(0, 0): [0], (1, 0): [1]},
    )


def test_layout_calibration_and_speed_energy_tradeoff():
    instance = _instance()
    instance.generate_agv_params(seed=123, transport_time_ratio=0.30)

    metadata = instance.extension_metadata
    assert metadata["energy_model_version"] == "calibrated_dimensionless_v5"
    assert metadata["transport_time_ratio_realized"] == pytest.approx(0.30)
    assert np.allclose(instance.machine_idle_power, instance.machine_proc_power / 4.0)
    assert np.allclose(instance.machine_setup_power, instance.machine_proc_power / 2.0)

    loaded = [instance.transport_energy(10.0, speed, loaded=True)
              for speed in instance.speed_levels]
    empty = [instance.transport_energy(10.0, speed, loaded=False)
             for speed in instance.speed_levels]
    assert loaded[0] < loaded[1] < loaded[2]
    assert empty[0] < empty[1] < empty[2]
    assert all(load > no_load for load, no_load in zip(loaded, empty))


def test_machine_idle_energy_uses_active_window_not_global_makespan():
    instance = _instance()
    instance.machine_proc_power = np.array([5.0, 5.0])
    instance.machine_idle_power = np.array([2.0, 2.0])
    instance.machine_setup_power = np.array([3.0, 3.0])
    instance.machine_setup_time = np.array([2.0, 2.0])
    schedule = SimpleNamespace(
        machine_schedule={
            0: [(0, 0, 10.0, 20.0), (1, 0, 30.0, 40.0)],
            1: [],
        }
    )

    details = compute_machine_energy(instance, schedule, makespan=100.0)

    # Active span 30 - processing 20 - one setup of 2 = 8 standby units.
    assert details[0]["idle_time"] == pytest.approx(8.0)
    assert details[0]["idle"] == pytest.approx(16.0)
    assert details[1]["idle_time"] == 0.0
    assert details[1]["total"] == 0.0


def test_nonpositive_transport_ratio_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        _instance().generate_agv_params(seed=1, transport_time_ratio=0.0)


def test_energy_initialization_minimizes_processing_energy_not_power_only():
    instance = SimpleNamespace(
        total_operations=1,
        num_jobs=1,
        num_operations=[1],
        num_agv=1,
        num_speeds=3,
        machine_proc_power=np.array([5.0, 8.0]),
        get_compatible_machines=lambda i, j: [0, 1],
        get_processing_time=lambda i, j, u: [10.0, 5.0][u],
    )

    chromosome = energy_chromosome(instance, np.random.RandomState(7))

    # Machine 1 has higher power but lower energy: 8*5 < 5*10.
    assert chromosome.ma.tolist() == [1]
