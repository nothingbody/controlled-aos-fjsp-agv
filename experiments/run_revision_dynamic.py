"""Expanded dynamic-rescheduling experiment for the revision.

The original dynamic experiment used one random instance. This revision script
extends it to multiple random instances, disruption types, intensities, methods,
and seeds while keeping row-level incremental output for long remote runs.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.loader import generate_random_instance
from experiments.exp_utils import ExperimentLogger, IncrementalSaver, Timer, estimate_remaining
from src.algorithm.dynamic.disruption_evaluator import evaluate_disruption
from src.algorithm.dynamic.rescheduler import GradedRescheduler
from src.algorithm.grl_ea import GRLEA
from src.algorithm.nsga3.decoding import evaluate
from src.algorithm.nsga3.encoding import Chromosome
from src.baselines.pure_nsga3 import PureNSGA3
from src.baselines.right_shift import FullRescheduler, RightShiftRescheduler
from src.environment.dynamic_events import DynamicEvent, EventGenerator, EventType


INIT_POP = 24
INIT_GEN = 24

FULL_POP = 24
FULL_GEN = 24

LOCAL_POP = 20
LOCAL_GEN = 24

ROLLING_POP = 24
ROLLING_GEN = 28

GRADED_LOCAL_POP = 20
GRADED_LOCAL_GEN = 24
GRADED_GLOBAL_POP = 28
GRADED_GLOBAL_GEN = 32


INSTANCE_SPECS = [
    ("Dyn01_10x5x2", 10, 5, 2, (3, 5), 6101),
    ("Dyn02_12x6x2", 12, 6, 2, (3, 5), 6102),
    ("Dyn03_15x8x3", 15, 8, 3, (3, 6), 6103),
    ("Dyn04_18x8x3", 18, 8, 3, (3, 6), 6104),
    ("Dyn05_20x10x3", 20, 10, 3, (3, 6), 6105),
    ("Dyn06_22x10x4", 22, 10, 4, (3, 6), 6106),
    ("Dyn07_25x12x4", 25, 12, 4, (3, 6), 6107),
    ("Dyn08_28x12x4", 28, 12, 4, (3, 6), 6108),
    ("Dyn09_30x15x5", 30, 15, 5, (3, 7), 6109),
    ("Dyn10_35x15x5", 35, 15, 5, (3, 7), 6110),
]

SCENARIOS = {
    "new_jobs": "New jobs",
    "machine_breakdown": "Machine breakdown",
    "agv_failure": "AGV failure",
    "mixed": "Mixed",
}

INTENSITIES = {
    "low": {
        "new_job_rate": 0.04,
        "machine_rate": 0.004,
        "agv_rate": 0.004,
        "max_events": 3,
        "machine_repair_range": (3, 10),
        "agv_repair_range": (2, 8),
    },
    "medium": {
        "new_job_rate": 0.08,
        "machine_rate": 0.010,
        "agv_rate": 0.010,
        "max_events": 5,
        "machine_repair_range": (5, 18),
        "agv_repair_range": (3, 12),
    },
    "high": {
        "new_job_rate": 0.14,
        "machine_rate": 0.020,
        "agv_rate": 0.020,
        "max_events": 8,
        "machine_repair_range": (8, 28),
        "agv_repair_range": (5, 20),
    },
}

METHODS = ["RightShift", "LocalRepair", "RollingHorizonEA", "FullReschedule", "GradedResponse"]
SCENARIO_INDEX = {name: idx for idx, name in enumerate(SCENARIOS)}
INTENSITY_INDEX = {name: idx for idx, name in enumerate(INTENSITIES)}


@dataclass(frozen=True)
class Task:
    instance_name: str
    num_jobs: int
    num_machines: int
    num_agv: int
    ops_range: tuple[int, int]
    instance_seed: int
    scenario: str
    intensity: str
    seed: int
    done_methods: tuple[str, ...]


def build_instance(task: Task):
    return generate_random_instance(
        task.num_jobs,
        task.num_machines,
        task.num_agv,
        ops_range=task.ops_range,
        compatible_range=(1, min(4, task.num_machines)),
        pt_range=(1, 20),
        seed=task.instance_seed,
    )


def generate_initial_schedule(instance, seed: int):
    solver = GRLEA(
        instance,
        pop_size=INIT_POP,
        max_gen=INIT_GEN,
        use_grl=False,
        seed=seed,
        verbose=False,
    )
    archive = solver.run()
    return min(archive, key=lambda c: c.objectives.makespan)


def clone_chromosome(chromosome: Chromosome, instance) -> Chromosome:
    return Chromosome(
        instance=instance,
        os=chromosome.os.copy(),
        ma=chromosome.ma.copy(),
        agv_assign=chromosome.agv_assign.copy(),
        agv_speed=chromosome.agv_speed.copy(),
    )


def add_new_job_to_instance(instance, event: DynamicEvent) -> None:
    if event.event_type != EventType.NEW_JOB:
        return
    while instance.num_jobs < event.new_job_id:
        instance.num_operations.append(0)
        instance.num_jobs += 1
    if instance.num_jobs == event.new_job_id:
        instance.num_jobs += 1
        instance.num_operations.append(event.new_num_ops)
    if event.new_processing_times:
        instance.processing_times.update(event.new_processing_times)
    if event.new_compatible_machines:
        instance.compatible_machines.update(event.new_compatible_machines)


def append_new_job_to_chromosome(chromosome: Chromosome, event: DynamicEvent, rng) -> Chromosome:
    child = chromosome.copy()
    new_entries = np.full(event.new_num_ops, event.new_job_id)
    child.os = np.concatenate([child.os, new_entries])

    new_ma = []
    for j in range(event.new_num_ops):
        machines = event.new_compatible_machines.get((event.new_job_id, j), [])
        if not machines:
            new_ma.append(0)
            continue
        times = [event.new_processing_times.get((event.new_job_id, j, u), float("inf")) for u in machines]
        new_ma.append(int(np.argmin(times)))
    child.ma = np.concatenate([child.ma, np.array(new_ma, dtype=int)])
    child.agv_assign = np.concatenate([
        child.agv_assign,
        rng.randint(0, child.instance.num_agv, size=event.new_num_ops),
    ])
    child.agv_speed = np.concatenate([
        child.agv_speed,
        np.ones(event.new_num_ops, dtype=int),
    ])
    child.invalidate()
    return child


class LocalRepairRescheduler:
    def __init__(self, instance, seed: int):
        self.instance = instance
        self.rng = np.random.RandomState(seed)
        self.inner = GradedRescheduler(
            instance,
            theta1=0.0,
            theta2=1.0,
            local_pop_size=LOCAL_POP,
            local_max_gen=LOCAL_GEN,
            global_pop_size=ROLLING_POP,
            global_max_gen=ROLLING_GEN,
            seed=seed,
        )
        self.response_log = []

    def respond(self, event, current_schedule, current_chromosome, current_time):
        d_value = evaluate_disruption(event, current_schedule, self.instance, current_time)
        base = current_chromosome
        if event.event_type == EventType.NEW_JOB:
            add_new_job_to_instance(self.instance, event)
            base = append_new_job_to_chromosome(current_chromosome, event, self.rng)
        result = self.inner._level2_local_reschedule(event, base, current_time)
        self.response_log.append({
            "time": current_time,
            "event_type": event.event_type.value,
            "disruption": d_value,
            "level": 2,
            "makespan_before": current_chromosome.objectives.makespan,
            "makespan_after": result.objectives.makespan,
        })
        return result


class RollingHorizonEARescheduler:
    def __init__(self, instance, seed: int):
        self.instance = instance
        self.rng = np.random.RandomState(seed)
        self.inner = GradedRescheduler(
            instance,
            theta1=-1.0,
            theta2=-0.5,
            local_pop_size=LOCAL_POP,
            local_max_gen=LOCAL_GEN,
            global_pop_size=ROLLING_POP,
            global_max_gen=ROLLING_GEN,
            seed=seed,
        )
        self.response_log = []

    def respond(self, event, current_schedule, current_chromosome, current_time):
        d_value = evaluate_disruption(event, current_schedule, self.instance, current_time)
        base = current_chromosome
        if event.event_type == EventType.NEW_JOB:
            add_new_job_to_instance(self.instance, event)
            base = append_new_job_to_chromosome(current_chromosome, event, self.rng)
        result = self.inner._level3_global_reschedule(event, base, current_time)
        self.response_log.append({
            "time": current_time,
            "event_type": event.event_type.value,
            "disruption": d_value,
            "level": 3,
            "makespan_before": current_chromosome.objectives.makespan,
            "makespan_after": result.objectives.makespan,
        })
        return result


class SafeGradedRescheduler(GradedRescheduler):
    def __init__(self, instance, seed: int):
        super().__init__(
            instance,
            theta1=0.15,
            theta2=0.40,
            local_pop_size=GRADED_LOCAL_POP,
            local_max_gen=GRADED_LOCAL_GEN,
            global_pop_size=GRADED_GLOBAL_POP,
            global_max_gen=GRADED_GLOBAL_GEN,
            seed=seed,
        )

    def respond(self, event, current_schedule, current_chromosome, current_time):
        if event.event_type != EventType.NEW_JOB:
            return super().respond(event, current_schedule, current_chromosome, current_time)

        d_value = evaluate_disruption(event, current_schedule, self.instance, current_time)
        add_new_job_to_instance(self.instance, event)
        base = append_new_job_to_chromosome(current_chromosome, event, self.rng)

        if d_value < self.theta1:
            level = 1
            result = base
        elif d_value < self.theta2:
            level = 2
            result = self._level2_local_reschedule(event, base, current_time)
        else:
            level = 3
            result = self._level3_global_reschedule(event, base, current_time)

        self.response_log.append({
            "time": current_time,
            "event_type": event.event_type.value,
            "disruption": d_value,
            "level": level,
            "makespan_before": current_chromosome.objectives.makespan,
            "makespan_after": result.objectives.makespan,
        })
        return result


class SafeRightShiftRescheduler(RightShiftRescheduler):
    def respond(self, event, current_schedule, current_chromosome, current_time):
        if event.event_type == EventType.NEW_JOB:
            add_new_job_to_instance(self.instance, event)
        return super().respond(event, current_schedule, current_chromosome, current_time)


def make_rescheduler(method: str, instance, seed: int):
    if method == "RightShift":
        return SafeRightShiftRescheduler(instance, seed=seed)
    if method == "LocalRepair":
        return LocalRepairRescheduler(instance, seed=seed)
    if method == "RollingHorizonEA":
        return RollingHorizonEARescheduler(instance, seed=seed)
    if method == "FullReschedule":
        return FullRescheduler(instance, pop_size=FULL_POP, max_gen=FULL_GEN, seed=seed)
    if method == "GradedResponse":
        return SafeGradedRescheduler(instance, seed=seed)
    raise ValueError(f"Unknown method: {method}")


def scenario_rates(scenario: str, intensity: str) -> dict:
    base = INTENSITIES[intensity]
    rates = {
        "new_job_rate": 0.0,
        "machine_breakdown_rate": 0.0,
        "agv_breakdown_rate": 0.0,
    }
    if scenario in ("new_jobs", "mixed"):
        rates["new_job_rate"] = base["new_job_rate"]
    if scenario in ("machine_breakdown", "mixed"):
        rates["machine_breakdown_rate"] = base["machine_rate"]
    if scenario in ("agv_failure", "mixed"):
        rates["agv_breakdown_rate"] = base["agv_rate"]
    return rates


def generate_events(instance, scenario: str, intensity: str, cmax: float, seed: int):
    base = INTENSITIES[intensity]
    rates = scenario_rates(scenario, intensity)
    generator = EventGenerator(
        instance,
        machine_repair_range=base["machine_repair_range"],
        agv_repair_range=base["agv_repair_range"],
        seed=seed,
        **rates,
    )
    events = generator.generate_events(time_horizon=cmax, start_time=0.2 * cmax)
    return events[: base["max_events"]]


def event_counts(events: list[DynamicEvent]) -> dict[str, int]:
    return {
        "new_job_events": sum(e.event_type == EventType.NEW_JOB for e in events),
        "machine_events": sum(e.event_type == EventType.MACHINE_BREAKDOWN for e in events),
        "agv_events": sum(e.event_type == EventType.AGV_BREAKDOWN for e in events),
    }


def summarize_levels(rescheduler) -> dict[str, int]:
    logs = getattr(rescheduler, "response_log", [])
    return {
        "level1_count": sum(r.get("level") == 1 for r in logs),
        "level2_count": sum(r.get("level") == 2 for r in logs),
        "level3_count": sum(r.get("level") == 3 for r in logs),
    }


def run_method(method: str, base_instance, init_chrom, events, task: Task) -> dict:
    method_instance = copy.deepcopy(base_instance)
    current = clone_chromosome(init_chrom, method_instance)
    ref_schedule = current.schedule
    cmax_before = current.objectives.makespan
    tec_before = current.objectives.total_energy

    response_times = []
    errors = []
    start = time.time()
    rescheduler = make_rescheduler(method, method_instance, seed=task.seed)

    for event in events:
        try:
            t0 = time.time()
            current = rescheduler.respond(
                event=event,
                current_schedule=current.schedule,
                current_chromosome=current,
                current_time=event.time,
            )
            response_times.append(time.time() - t0)
        except Exception as exc:
            errors.append(f"{event.event_type.value}:{type(exc).__name__}:{exc}")
            break

    obj = current.objectives
    stability_obj = evaluate(current, ref_schedule=ref_schedule)
    finite = all(math.isfinite(v) for v in [obj.makespan, obj.total_energy, stability_obj.stability])
    counts = event_counts(events)
    levels = summarize_levels(rescheduler)

    return {
        "instance": task.instance_name,
        "num_jobs": task.num_jobs,
        "num_machines": task.num_machines,
        "num_agv": task.num_agv,
        "scenario": task.scenario,
        "scenario_name": SCENARIOS[task.scenario],
        "intensity": task.intensity,
        "seed": task.seed,
        "method": method,
        "num_events": len(events),
        **counts,
        "cmax_before": cmax_before,
        "cmax_after": obj.makespan,
        "cmax_deviation_pct": (obj.makespan - cmax_before) / max(cmax_before, 1e-9) * 100.0,
        "tec_before": tec_before,
        "tec_after": obj.total_energy,
        "tec_deviation_pct": (obj.total_energy - tec_before) / max(tec_before, 1e-9) * 100.0,
        "stability_f4": stability_obj.stability,
        "avg_response_time": float(np.mean(response_times)) if response_times else 0.0,
        "total_response_time": float(np.sum(response_times)) if response_times else 0.0,
        "wall_time": time.time() - start,
        "success": int(finite and not errors),
        **levels,
        "error": " | ".join(errors),
    }


def run_task(task: Task) -> list[dict]:
    base_instance = build_instance(task)
    init_chrom = generate_initial_schedule(base_instance, task.seed)
    event_seed = (
        task.instance_seed * 1000
        + task.seed * 17
        + SCENARIO_INDEX[task.scenario] * 101
        + INTENSITY_INDEX[task.intensity] * 1009
    )
    events = generate_events(base_instance, task.scenario, task.intensity, init_chrom.objectives.makespan, event_seed)

    records = []
    done = set(task.done_methods)
    for method in METHODS:
        if method in done:
            continue
        records.append(run_method(method, base_instance, init_chrom, events, task))
    return records


def parse_list(value: str, valid: Iterable[str]) -> list[str]:
    valid = list(valid)
    if value.lower() == "all":
        return valid
    selected = [v.strip() for v in value.split(",") if v.strip()]
    invalid = sorted(set(selected) - set(valid))
    if invalid:
        raise ValueError(f"Invalid values {invalid}; valid values are {valid}")
    return selected


def completed_methods(csv_path: str) -> dict[tuple, set[str]]:
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    out: dict[tuple, set[str]] = {}
    required = {"instance", "scenario", "intensity", "seed", "method"}
    if not required.issubset(df.columns):
        return {}
    for row in df.itertuples(index=False):
        key = (row.instance, row.scenario, row.intensity, int(row.seed))
        out.setdefault(key, set()).add(row.method)
    return out


def build_tasks(args, done_map) -> list[Task]:
    selected_instances = INSTANCE_SPECS[: args.instances]
    scenarios = parse_list(args.scenarios, SCENARIOS.keys())
    intensities = parse_list(args.intensities, INTENSITIES.keys())
    seeds = list(range(args.seed_start, args.seed_end))
    tasks = []
    for name, n_jobs, n_machines, n_agv, ops_range, inst_seed in selected_instances:
        for scenario in scenarios:
            for intensity in intensities:
                for seed in seeds:
                    key = (name, scenario, intensity, seed)
                    done_methods = tuple(sorted(done_map.get(key, set())))
                    if set(done_methods) >= set(METHODS):
                        continue
                    tasks.append(Task(name, n_jobs, n_machines, n_agv, ops_range, inst_seed,
                                      scenario, intensity, seed, done_methods))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/revision/dynamic_extended")
    parser.add_argument("--result-file", default="results/revision/dynamic_extended/dynamic_extended_runs.csv")
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--intensities", default="all")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--seed-end", type=int, default=52)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    log = ExperimentLogger("revision_dynamic_extended", log_dir="results/revision/logs")
    saver = IncrementalSaver(args.result_file)
    timer = Timer()

    done_map = completed_methods(args.result_file)
    tasks = build_tasks(args, done_map)
    total_task_count = len(tasks)
    total_rows = total_task_count * len(METHODS)

    log.section("Revision Dynamic Extended Experiment")
    log.info(f"Result file: {args.result_file}")
    log.info(f"Existing rows: {saver.count}")
    log.info(f"Pending tasks: {total_task_count}, approximate pending rows: {total_rows}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Budgets: init={INIT_POP}x{INIT_GEN}, full={FULL_POP}x{FULL_GEN}, "
             f"local={LOCAL_POP}x{LOCAL_GEN}, rolling={ROLLING_POP}x{ROLLING_GEN}, "
             f"graded_global={GRADED_GLOBAL_POP}x{GRADED_GLOBAL_GEN}")

    completed_tasks = 0
    errors = 0

    def handle_result(records):
        nonlocal completed_tasks, errors
        completed_tasks += 1
        for record in records:
            if record.get("error"):
                errors += 1
            saver.append(record)
        eta = estimate_remaining(completed_tasks, total_task_count, time.time() - timer.start_time)
        if completed_tasks == 1 or completed_tasks % max(1, min(20, total_task_count // 20 or 1)) == 0:
            log.info(f"completed_tasks={completed_tasks}/{total_task_count} rows={saver.count} "
                     f"errors={errors} ETA={eta}")

    if args.workers <= 1:
        for task in tasks:
            handle_result(run_task(task))
    else:
        with Pool(processes=args.workers) as pool:
            for records in pool.imap_unordered(run_task, tasks, chunksize=1):
                handle_result(records)

    log.info(f"Done. rows={saver.count}, errors={errors}, elapsed={timer.elapsed()}")
    log.info(f"Results: {args.result_file}")


if __name__ == "__main__":
    main()
