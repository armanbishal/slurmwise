# Knowledge base

Each file here is one **failure family**: a cluster failure mode the classifier can detect,
with its symptoms, causes, fixes, and how to confirm. The agent embeds these into Chroma
and retrieves the matching one(s) when it diagnoses a job (retrieval-augmented generation).

## Schema

Front matter:

```yaml
---
id: <stable_snake_case_id>  
title: <human title>
kinds: [<classifier kinds this doc answers>]
tags: [<keywords and error strings for retrieval>]
severity: user | infra | mixed
---
```

Body sections, in this order:

```markdown
# <title>

## Symptoms      - how it shows up in sacct state and the logs
## Causes        - why it usually happens
## Fixes         - numbered, concrete, with exact flags / env vars
## Verify        - one or two commands to confirm the cause or check the fix
```

## Adding an entry

1. Copy the schema above into `new_id.md`.
2. Fill the sections; put the exact error strings in `tags` so retrieval finds them.
3. Run `slurmpilot kb build` to re-index.
4. Add a labeled example to `../benchmark/labeled_failures.json` and run
   `slurmpilot eval` to check the classifier and retriever still find it.

## Current coverage

These are the common families, not every possible SLURM failure. The base is designed to
grow: QOS/account limits, allocation exhaustion, dependency chains, and MPI bootstrap
errors are natural next additions.