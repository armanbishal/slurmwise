---
id: time_limit
title: Wall-clock time limit
kinds: [time_limit]
tags: [--time, TIMEOUT, DUE TO TIME LIMIT, checkpoint, requeue, array job]
severity: user
---
# Wall-clock time limit

## Symptoms
- `*** JOB N ON node CANCELLED AT ... DUE TO TIME LIMIT ***`
- sacct state `TIMEOUT`, Elapsed equals Timelimit

## Causes
- The job simply needed more wall time than requested
- A sweep or benchmark run serially that should have been parallel
- No checkpointing, so a long job cannot resume

## Fixes
1. Ask for more time: `#SBATCH --time=12:00:00`. Check the cap with `sinfo -o "%P %l"`;
   move to a longer partition if needed.
2. Checkpoint and requeue: save every N steps, add `#SBATCH --signal=B:USR1@300`, trap
   USR1 to save and `scontrol requeue $SLURM_JOB_ID`.
3. Estimate from the log ETA: a job at 95% when killed needs ~10% more time; one at 40%
   needs a different plan (more workers, bigger batch, fewer repeats).
4. Split sweeps into array jobs: `#SBATCH --array=0-9`, shard by `$SLURM_ARRAY_TASK_ID`.

## Verify
- `sacct -j <id> --format=JobID,Elapsed,Timelimit` to confirm Elapsed hit the limit.
