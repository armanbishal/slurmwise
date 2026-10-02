---
id: qos_limit
title: Blocked by a QOS or account limit
kinds: [qos_limit]
tags: [QOSMaxJobsPerUserLimit, AssocMaxJobsLimit, InvalidQOS, InvalidAccount, QOS, --qos, --account, pending]
severity: user
---
# Blocked by a QOS or account limit

## Symptoms
- Job stays PENDING with Reason `QOSMaxJobsPerUserLimit`, `AssocMaxJobsLimit`, or similar
- sbatch rejects it: `Invalid qos specification` or `Invalid account`
- `Job violates accounting/QOS policy`

## Causes
- You already have the max number of running/queued jobs this QOS allows
- Wrong `--qos` or `--account` for your allocation
- The QOS caps per-user CPUs, GPUs, or submit count and you exceeded it

## Fixes
1. Check your limits: `sacctmgr show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU`.
2. Use a valid account/QOS: `sacctmgr show assoc user=$USER format=Account,QOS`.
3. Wait for running jobs to finish, or submit fewer at once; use array jobs with
   `%N` to cap concurrency (`--array=0-99%10`).

## Verify
- `squeue -u $USER --start` shows the Reason; `scontrol show job <id>` gives the full reason.
