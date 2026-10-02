---
id: node_fail
title: Node failure or preemption
kinds: [node_fail, preempted]
tags: [NODE_FAIL, PREEMPTED, requeue, --requeue, hardware, preemptible]
severity: infra
---
# Node failure or preemption

## Symptoms
- `*** JOB N ON node CANCELLED AT ... DUE TO NODE FAILURE ***`
- sacct state `NODE_FAIL` or `PREEMPTED`

## Causes
- The compute node crashed or was drained while the job ran (not your code)
- The job was on a preemptible partition/QOS and got preempted by higher priority

## Fixes
1. Resubmit. Resume from the last checkpoint if you have one.
2. Add `#SBATCH --requeue` so SLURM resubmits automatically next time.
3. If the same node keeps dying, `#SBATCH --exclude=<node>` and open a ticket.
4. For preemption, use a non-preemptible partition/QOS for work that cannot checkpoint.

## Verify
- `sacct -j <id> --format=JobID,State,Reason` and the slurmctld log confirm it was the node.
