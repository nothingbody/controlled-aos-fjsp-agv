import numpy as np

from src.algorithm.nsga3.encoding import Chromosome
from src.algorithm.nsga3.decoding import evaluate
from src.problem.energy_model import compute_agv_energy
from src.problem.instance import FJSPAGVInstance


def test_empty_travel_is_included_in_total_agv_energy():
    instance = FJSPAGVInstance(
        num_jobs=2,
        num_machines=2,
        num_agv=1,
        num_operations=[1, 1],
        processing_times={(0, 0, 0): 2.0, (1, 0, 1): 2.0},
        compatible_machines={(0, 0): [0], (1, 0): [1]},
        distance_matrix=np.array(
            [
                [0.0, 5.0, 9.0],
                [5.0, 0.0, 4.0],
                [9.0, 4.0, 0.0],
            ]
        ),
        speed_levels=[1.0],
        machine_proc_power=np.zeros(2),
        machine_idle_power=np.zeros(2),
        machine_setup_power=np.zeros(2),
        machine_setup_time=np.zeros(2),
        agv_load_power_params=(0.0, 0.0, 2.0),
        agv_empty_power_params=(0.0, 0.0, 1.0),
    )
    chromosome = Chromosome(
        os=np.array([0, 1]),
        ma=np.array([0, 0]),
        agv_assign=np.array([0, 0]),
        agv_speed=np.array([0, 0]),
        instance=instance,
    )

    objectives = evaluate(chromosome)
    detailed = compute_agv_energy(instance, chromosome.schedule)

    # Loaded: depot->M1 (5) and depot->M2 (9), power 2 => 28.
    # Empty: M1->depot before the second task (5), power 1 => 5.
    assert detailed[0]["load"] == 28.0
    assert detailed[0]["empty"] == 5.0
    assert detailed[0]["total"] == 33.0
    assert objectives.total_energy == 33.0
