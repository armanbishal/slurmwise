---
id: general
title: General triage (unknown or uncategorized failure)
kinds: [unknown, python_exception, segfault, cancelled]
tags: [triage, sacct, exit code, seff, Segmentation fault, core dumped, Traceback]
severity: mixed
---
# General triage (unknown or uncategorized failure)

## Symptoms
- A failure with no signature the classifier recognizes
- A bare Python traceback, a segfault/`core dumped`, or a plain `CANCELLED`

## Causes
- An application bug specific to your code (read the last exception)
- Segfault: mixed CUDA libraries, a broken wheel, or a C extension built on another node
- CANCELLED with no reason: QOS/account limits, or someone cancelled it

## Fixes
1. Read the end of stderr first; the last exception is the one that matters.
2. Decode the exit code: `0:9` SIGKILL (often OOM), `0:15` SIGTERM, `1:0` program exited 1.
3. Segfault: try a clean env or pin the node type with `--constraint`.
4. CANCELLED: check `sacct ... --format=Reason` and `sacctmgr show assoc user=$USER`.
5. No log at all: the output path probably did not exist; SLURM drops it silently.

## Verify
- `sacct -j <id> --format=JobID,State,ExitCode,Reason` and `seff <id>` for a summary.
