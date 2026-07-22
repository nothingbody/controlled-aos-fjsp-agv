"""FJSP-AGV 问题实例数据结构"""

from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class FJSPAGVInstance:
    """FJSP-AGV问题实例

    Attributes:
        num_jobs: 工件数 n
        num_machines: 机器数 m
        num_agv: AGV数 k
        num_operations: 每个工件的工序数列表 [n_1, n_2, ..., n_n]
        processing_times: 加工时间字典 {(i, j, u): p_iju}
        compatible_machines: 每道工序的可选机器列表 {(i, j): [u1, u2, ...]}
        distance_matrix: 机器间距离矩阵 (m+1) x (m+1)，索引0为装载/卸载站
        speed_levels: AGV速度档位列表
        machine_proc_power: 各机器加工功率 [P1, P2, ..., Pm]
        machine_idle_power: 各机器空载功率
        machine_setup_power: 各机器换模功率
        machine_setup_time: 各机器换模时间
        agv_load_power_params: AGV载货功率模型参数 (alpha, beta, gamma)
        agv_empty_power_params: AGV空载功率模型参数
    """
    num_jobs: int
    num_machines: int
    num_agv: int = 3
    num_operations: list = field(default_factory=list)
    processing_times: dict = field(default_factory=dict)
    compatible_machines: dict = field(default_factory=dict)
    distance_matrix: Optional[np.ndarray] = None
    speed_levels: list = field(default_factory=lambda: [0.5, 0.75, 1.0])
    machine_proc_power: Optional[np.ndarray] = None
    machine_idle_power: Optional[np.ndarray] = None
    machine_setup_power: Optional[np.ndarray] = None
    machine_setup_time: Optional[np.ndarray] = None
    agv_load_power_params: tuple = (5.0, 0.0, 0.0)
    agv_empty_power_params: tuple = (3.0, 0.0, 0.0)
    layout_coordinates: Optional[np.ndarray] = None
    extension_metadata: dict = field(default_factory=dict)

    @property
    def total_operations(self) -> int:
        return sum(self.num_operations)

    @property
    def num_speeds(self) -> int:
        return len(self.speed_levels)

    def get_processing_time(self, job: int, op: int, machine: int) -> float:
        return self.processing_times.get((job, op, machine), float('inf'))

    def get_compatible_machines(self, job: int, op: int) -> list:
        return self.compatible_machines.get((job, op), [])

    def get_distance(self, from_machine: int, to_machine: int) -> float:
        """获取两台机器之间的距离（索引0为装载/卸载站）"""
        if self.distance_matrix is None:
            return 0.0
        return self.distance_matrix[from_machine, to_machine]

    def agv_load_power(self, speed: float) -> float:
        """计算AGV载货功率 P_load(v) = alpha*v^2 + beta*v + gamma"""
        a, b, c = self.agv_load_power_params
        return a * speed ** 2 + b * speed + c

    def agv_empty_power(self, speed: float) -> float:
        """计算AGV空载功率"""
        a, b, c = self.agv_empty_power_params
        return a * speed ** 2 + b * speed + c

    def transport_energy(self, distance: float, speed: float, loaded: bool = True) -> float:
        """计算单次运输能耗 = 功率 × 时间"""
        if distance <= 0 or speed <= 0:
            return 0.0
        power = self.agv_load_power(speed) if loaded else self.agv_empty_power(speed)
        time = distance / speed
        return power * time

    def generate_agv_params(self, seed: int = 42,
                            transport_time_ratio: float = 0.30):
        """Generate a deterministic, scale-calibrated synthetic extension.

        Raw unit-square coordinates are scaled so that the median trip at the
        middle speed equals ``transport_time_ratio`` times the median eligible
        processing time.  The original FJSP files contain no layout or energy
        fields, so all generated values are reported as a normalized index.
        """
        if transport_time_ratio <= 0:
            raise ValueError("transport_time_ratio must be positive")
        rng = np.random.RandomState(seed)

        size = self.num_machines + 1
        unit_coords = rng.uniform(0.0, 1.0, size=(size, 2))
        raw_distances = np.linalg.norm(
            unit_coords[:, None, :] - unit_coords[None, :, :], axis=-1
        )
        nonzero_distances = raw_distances[np.triu_indices(size, k=1)]
        raw_median_distance = float(np.median(nonzero_distances))
        processing_values = np.asarray(
            [value for value in self.processing_times.values()
             if np.isfinite(value) and value > 0],
            dtype=float,
        )
        median_processing_time = (
            float(np.median(processing_values))
            if processing_values.size else 1.0
        )
        middle_speed = float(np.median(np.asarray(self.speed_levels, dtype=float)))
        distance_scale = (
            transport_time_ratio * median_processing_time * middle_speed
            / max(raw_median_distance, 1e-12)
        )
        self.layout_coordinates = unit_coords * distance_scale
        self.distance_matrix = raw_distances * distance_scale
        np.fill_diagonal(self.distance_matrix, 0)

        # Source-inspired power ratios in normalized energy units.
        base_power = rng.uniform(5.0, 10.0, self.num_machines)
        self.machine_proc_power = base_power
        self.machine_idle_power = base_power / 4.0
        self.machine_setup_power = base_power / 2.0
        self.machine_setup_time = (
            rng.uniform(0.05, 0.15, self.num_machines)
            * median_processing_time
        )

        loaded_coefficient = float(rng.uniform(4.0, 6.0))
        self.agv_load_power_params = (loaded_coefficient, 0.0, 0.0)
        self.agv_empty_power_params = (0.6 * loaded_coefficient, 0.0, 0.0)
        realized_median_distance = float(np.median(
            self.distance_matrix[np.triu_indices(size, k=1)]
        ))
        self.extension_metadata = {
            "energy_model_version": "calibrated_dimensionless_v5",
            "transport_time_ratio_target": float(transport_time_ratio),
            "transport_time_ratio_realized": float(
                realized_median_distance
                / max(middle_speed * median_processing_time, 1e-12)
            ),
            "median_processing_time": median_processing_time,
            "middle_speed": middle_speed,
            "layout_distance_scale": float(distance_scale),
            "machine_idle_boundary": "first_processing_start_to_last_completion",
            "energy_unit": "normalized_energy_index",
        }

    def summary(self) -> str:
        return (
            f"FJSP-AGV Instance: {self.num_jobs} jobs × {self.num_machines} machines "
            f"× {self.num_agv} AGVs, {self.total_operations} total operations"
        )
