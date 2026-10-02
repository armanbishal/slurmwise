---
id: nccl
title: NCCL / multi-node distributed error
kinds: [nccl]
tags: [nccl, NCCL error, ncclSystemError, NCCL WARN, torchrun, distributed, ib0, NCCL_SOCKET_IFNAME, DistBackendError, timeout]
severity: mixed
---
# NCCL / multi-node distributed error

## Symptoms
- `torch.distributed.DistBackendError: NCCL error ... unhandled system error`
- `NCCL WARN socketStartConnect ... No route to host` or `Connection refused`
- A watchdog timeout in a collective (ALLREDUCE), or a hang in `init_process_group`

## Causes
- Ranks cannot reach each other: wrong interface (eth instead of IB), firewall, or a
  master address that does not resolve on the other node
- `MASTER_PORT` collision with another job on the node
- World size / rank env mismatched across nodes (launcher bug)
- Different NCCL/CUDA versions in the env on different nodes

## Fixes
1. Pin the interface: `export NCCL_SOCKET_IFNAME=ib0` (check `ip link`); set
   `NCCL_IB_HCA=mlx5` if InfiniBand is present.
2. Set the master from SLURM:
   `MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)`, random free port.
3. Launch with `srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$GPUS ...`.
4. If one node keeps failing, `--exclude` it and tell the admins.

## Verify
- Re-run with `export NCCL_DEBUG=INFO` to see which step fails.
