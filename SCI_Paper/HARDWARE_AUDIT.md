# Hardware-use audit

Audit date: 2026-07-20

## Conclusion

The previously reported revision experiments were CPU runs, not GPU runs.

## Code evidence

- `experiments/run_static.py` constructs both GRL-enabled and non-GRL solvers with `device='cpu'`.
- The earlier and corrected `experiments/run_revision_aos.py` construct every selector with `device='cpu'`.
- `experiments/run_revision_sensitivity.py` uses `device="cpu"`.
- `experiments/run_revision_operator_ablation.py` uses `device="cpu"`.
- `results/revision/README_revision.md` describes 40 CPU workers for most revision reruns and 80 CPU workers for the budget diagnostic.
- `方案C_换机重跑指南.md` explicitly states that the PPO network is small, GPU speedup is limited, and CPU process parallelism was preferred.

PyTorch and CUDA may have been installed and CUDA may have been detectable, but the relevant experiment entry points overrode automatic device selection and passed CPU explicitly. Population decoding, objective evaluation, nondominated sorting, and most of the workload are NumPy/Python CPU operations in any case.

## v5 rerun policy

The v5 runner records `Execution_device=cpu` and `Torch_CUDA_available` in every output row. The latter is environmental metadata only and must not be interpreted as evidence that a run used the GPU. The GPUHome server is useful primarily for its CPU quota and memory; moving the small PPO network to CUDA while retaining multiprocess CPU decoding is not expected to reduce total runtime and can create GPU contention across worker processes.

## Resubmission server record

The v4 pilot was deployed on 2026-07-20 to the GPUHome container `jupyter-zsa6mvn7e9c93t6m` and then paused after its validity audit. The host exposes 104 logical processors to `nproc`, but the container cgroup fixes `cpu.max` at `1200000 100000`, corresponding to an effective quota of 12 CPU cores. The container reports 371 GiB RAM. Although the image contains PyTorch 2.11.0+cu128, no NVIDIA device node or `nvidia-smi` executable is exposed and `torch.cuda.is_available()` is false. The v5 smoke and formal experiments will therefore use 12 single-threaded CPU worker processes, with server metadata re-exported under the v5 result directory.
