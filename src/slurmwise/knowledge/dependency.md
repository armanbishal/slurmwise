---
id: dependency
title: Job dependency never satisfied
kinds: [dependency]
tags: [DependencyNeverSatisfied, --dependency, afterok, afterany, pending, dependency]
severity: user
---
# Job dependency never satisfied

## Symptoms
- Job stays PENDING with Reason `(Dependency)`, then `DependencyNeverSatisfied`
- It never starts and is eventually cancelled by the scheduler

## Causes
- `--dependency=afterok:<id>` but the prerequisite job failed, so the ok condition
  can never be met
- A typo'd or already-finished job id in the dependency
- A chain where an upstream job was cancelled

## Fixes
1. Check the prerequisite's final state: `sacct -j <dep_id> --format=JobID,State`.
2. If it failed, fix and resubmit it, then resubmit this job with a new dependency.
3. Use `afterany` instead of `afterok` if this job should run regardless of outcome.
4. Remove the dependency and submit manually once the prerequisite is done.

## Verify
- `scontrol show job <id>` shows the Dependency field and which job it waits on.
