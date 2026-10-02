---
id: allocation_exhausted
title: Allocation or billing balance exhausted
kinds: [allocation_exhausted]
tags: [AssocGrpBillingMinutes, service units, SUs, allocation, balance, AssocGrpCPUMinutesLimit, pending]
severity: user
---
# Allocation or billing balance exhausted

## Symptoms
- Job pends with Reason `AssocGrpBillingMinutes` or `AssocGrpCPUMinutesLimit`
- Submission rejected for being out of service units / allocation balance
- Common on allocation-based clusters (e.g. NCSA Delta) near the end of a grant period

## Causes
- The project's compute allocation (service units) is used up
- A single job requests more SUs than remain in the balance

## Fixes
1. Check the balance with your site's accounting tool (e.g. `accounts` on Delta).
2. Request a smaller/shorter job that fits the remaining balance.
3. Switch to another account you belong to: `#SBATCH --account=<other>`.
4. Request a supplement or renewal of the allocation.

## Verify
- `sacctmgr show assoc user=$USER format=Account,GrpTRESMins` and the site balance tool.
