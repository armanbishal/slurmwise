---
id: oom
title: Host memory (RAM) out of memory
kinds: [oom]
tags: [memory, cgroup, oom-kill, --mem, MaxRSS, Killed process, MemoryError, OUT_OF_MEMORY]
severity: user
---
# Host memory (RAM) out of memory

## Symptoms
- `slurmstepd: error: Detected N oom-kill event(s)` in the logs
- sacct state `OUT_OF_MEMORY`, exit code often `0:125` or `0:9`
- MaxRSS from sacct is within a few percent of ReqMem

## Causes
- Dataset loaded fully into RAM (pandas, `load_dataset` without streaming)
- DataLoader `num_workers` too high: each worker copies the dataset object
- Model loaded on CPU in fp32 before moving to the GPU
- Keeping both raw and tokenized copies of the data in memory

## Fixes
1. Raise the request: `#SBATCH --mem=64G` (or `--mem-per-cpu`). Check the partition cap
   with `scontrol show partition <name>` (MaxMemPerNode).
2. Drop `num_workers` to 2-4, or set `persistent_workers=False`.
3. Load the model with `low_cpu_mem_usage=True` / `device_map="cuda"`.
4. Stream or memory-map the dataset instead of loading it all.

## Verify
- `sacct -j <id> --format=JobID,MaxRSS,ReqMem` to see how close you were.
