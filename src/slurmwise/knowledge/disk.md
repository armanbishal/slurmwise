---
id: disk_quota
title: Disk quota, no space, or permission error
kinds: [disk_quota, permission, file_not_found]
tags: [quota, Errno 122, Errno 28, Errno 13, Errno 2, scratch, HF_HOME, Disk quota exceeded, No space left, Permission denied]
severity: user
---
# Disk quota, no space, or permission error

## Symptoms
- `OSError: [Errno 122] Disk quota exceeded` or `[Errno 28] No space left on device`
- `PermissionError: [Errno 13] Permission denied`
- `FileNotFoundError: [Errno 2] No such file or directory`

## Causes
- Home directory is small; caches fill it (`~/.cache/huggingface`, `~/.cache/torch`, `wandb/`)
- Writing outputs or checkpoints to home instead of scratch
- A shared or read-only path on compute nodes (permission)
- A relative path that only resolves from the login node's cwd (not found)

## Fixes
1. Quota/space: move caches to scratch in the script:
   `export HF_HOME=/scratch/$USER/hf; export TORCH_HOME=/scratch/$USER/torch`.
   Prune old checkpoints with `save_total_limit=2`.
2. Permission: write to your own scratch, or fix the group on the shared dir.
3. Not found: use absolute paths or `#SBATCH --chdir=`; check `$SLURM_SUBMIT_DIR`.

## Verify
- `quota -s` or `df -h $HOME`; `du -sh ~/.cache/*` to find the offender.
