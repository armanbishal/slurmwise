---
id: resources_unavailable
title: Requested resources never became available
kinds: [resources_unavailable]
tags: [ReqNodeNotAvail, Nodes required are DOWN, PartitionNodeLimit, constraint, --gres, --constraint, pending]
severity: mixed
---
# Requested resources never became available

## Symptoms
- Job pends with Reason `ReqNodeNotAvail`, `Nodes required for job are DOWN/DRAINED`
- `Requested node configuration is not available` at submit time
- A `--gres` or `--constraint` request no partition can satisfy

## Causes
- Asked for a GPU type, node count, or feature the partition does not have
- The matching nodes are down, drained, or reserved
- Time limit exceeds the partition maximum

## Fixes
1. Check what the partition offers: `sinfo -o "%P %l %D %G %f"`.
2. Match the request to a real GPU/feature name, or move to a partition that has it.
3. Lower the node count or time to fit the partition limits.
4. If nodes are temporarily down, wait or pick another partition.

## Verify
- `sinfo -p <partition> -o "%P %a %D %T %G"` shows node states and available GRES.
