"""Export immutable source and synthetic-extension manifests for resubmission."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_benchmark_set


DATASETS = {
    "Brandimarte": "data/benchmarks/brandimarte",
    "Hurink_edata": "data/benchmarks/hurink_edata",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serializable_array(value):
    return np.asarray(value, dtype=float).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/resubmission/manifests")
    parser.add_argument("--extension-seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for dataset, directory in DATASETS.items():
        instances = load_benchmark_set(
            directory, num_agv=3, extension_seed=args.extension_seed
        )
        for filename, instance in instances:
            source = Path(directory) / filename
            records.append(
                {
                    "dataset": dataset,
                    "instance": filename,
                    "source_path": source.as_posix(),
                    "source_sha256": sha256(source),
                    "source_bytes": source.stat().st_size,
                    "num_jobs": instance.num_jobs,
                    "num_machines": instance.num_machines,
                    "num_operations": instance.total_operations,
                    "num_agv": instance.num_agv,
                    "speed_levels": list(instance.speed_levels),
                    "extension_base_seed": args.extension_seed,
                    "extension_instance_seed": int(instance.extension_seed),
                    "layout_coordinates": serializable_array(
                        instance.layout_coordinates
                    ),
                    "distance_matrix": serializable_array(instance.distance_matrix),
                    "machine_proc_power": serializable_array(instance.machine_proc_power),
                    "machine_idle_power": serializable_array(instance.machine_idle_power),
                    "machine_setup_power": serializable_array(instance.machine_setup_power),
                    "machine_setup_time": serializable_array(instance.machine_setup_time),
                    "agv_load_power_params": list(instance.agv_load_power_params),
                    "agv_empty_power_params": list(instance.agv_empty_power_params),
                    "extension_metadata": dict(instance.extension_metadata),
                }
            )

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "instance_count": len(records),
    }

    with (out_dir / "benchmark_and_extension_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)
    with (out_dir / "environment_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(environment, stream, indent=2, ensure_ascii=False)

    print(
        f"exported {len(records)} instances to "
        f"{out_dir / 'benchmark_and_extension_manifest.json'}"
    )


if __name__ == "__main__":
    main()
