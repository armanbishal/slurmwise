<h1 align="center">SlurmWise: An LLM Agent for HPC Job Monitoring</h1>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Framework" src="https://img.shields.io/badge/LangGraph-MCP-black">
  <img alt="Vector DB" src="https://img.shields.io/badge/Chroma-RAG-orange">
  <img alt="Status" src="https://img.shields.io/badge/cluster-tested-success">
</p>

**Motivation:** On a shared HPC cluster, jobs run unattended and fail overnight. SLURM
hands you a job ID and a one-word state like `FAILED`, nothing more. Finding out why means
manually running `sacct`, opening the log, scrolling to the error, recognizing it, recalling
the fix, and editing the submit script, for every failure. And a failed job is not free: it
already consumed the GPU/CPU time, and the energy, it was allocated. At research scale
(hundreds of training and benchmarking jobs) this is a daily tax and a real source of wasted
compute.

**SlurmPilot** is a read-only assistant that automates both halves of that: diagnosing
failures, and quantifying what they cost.

- **Diagnosis** — explains why a job failed and suggests a fix. A rule-based classifier
  identifies the cause from the job record and logs (so it is reproducible, never
  hallucinated); a **Chroma** knowledge base supplies grounded fix notes via retrieval
  (RAG); and an optional LLM writes the explanation. The whole thing is served through a
  **Model Context Protocol (MCP)** server and orchestrated with a **LangGraph** agent, so
  the tools are reusable by any MCP client.
- **Analytics** — `slurmpilot stats` turns your job history into a failure rate, wasted
  GPU/CPU-hours, and the **energy that waste cost**, broken down
  by failure type and partition.

On a labeled benchmark of 31 real failure logs spanning 13
failure families, the classifier reaches **97% accuracy** and the retriever finds the
correct fix document with **Recall@3 of 0.97** (Recall@1 0.94, MRR 0.95). Run it yourself
with `slurmpilot eval`.

**Deployed on NCSA Delta.** On a real 30-day window it measured a 17% failure rate and
~24 wasted GPU-hours (about 15 kWh of energy) from timed-out jobs, concentrated on one
partition.

## Quick start

```bash
# install (Python 3.10+)
pip install -e ".[embed]"

# fleet analytics over your job history
slurmpilot stats

# diagnose one failed job
slurmpilot diagnose <jobid> --no-llm --direct

# list your failed jobs to get ids
slurmpilot jobs --state FAILED,TIMEOUT,OUT_OF_MEMORY

# score the diagnosis pipeline on the labeled benchmark
slurmpilot eval
```

No GPU or model required. Everything is read-only: it calls only `sacct`, `scontrol`,
`sinfo` and reads log files.

* * *

## Requirements

**System**
- A SLURM cluster login node with `sacct`, `scontrol`, and `sinfo` on `PATH` (for the real
  backend). Without them, a bundled mock backend runs anywhere.
- Python **3.10 or newer**.

**Python dependencies** (installed automatically)
- `mcp`, `langgraph`, `langchain-core`, `langchain-openai`
- `chromadb`, optional `sentence-transformers` for embeddings
- `typer`, `rich`, `pyyaml`

**Optional**
- Ollama

Verify after install:

```bash
slurmpilot --help
pytest
```

* * *

## Repository Layout

```
slurmpilot/
├── slurmpilot/
│   ├── backends/
│   │   ├── base.py          # JobInfo record
│   │   ├── real.py          # shells out to sacct / scontrol / sinfo
│   │   └── mock.py          # fixture-driven backend
│   ├── classify.py          # rule-based failure classifier (regex + sacct state)
│   ├── kb.py                # knowledge base: markdown -> Chroma, query
│   ├── knowledge/           # failure-pattern fix notes
│   ├── patch.py             # turns a diagnosis into an sbatch diff
│   ├── mcp_server.py        # exposes SLURM as read-only MCP tools
│   ├── agent/
│   │   ├── graph.py         # LangGraph implementation
│   │   ├── tools.py         # MCP (stdio) and in-process tool clients
│   │   ├── llm.py           # Ollama
│   │   └── chat.py          # Chat agent
│   └── cli.py               # cmd
├── tests/                   # runs on mock fixtures
└── README.md
```

* * *

## Reproducing the Results

### Stage 0 — Install

On a laptop:

```bash
cd slurmpilot
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[embed,dev]"
```

On a cluster whose default `python3` is older than 3.10 (e.g. NCSA Delta, which ships 3.9),
use the bundled miniforge to make a 3.11 environment with working SSL:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n slurmpilot python=3.11 openssl ca-certificates pip setuptools wheel
conda activate slurmpilot
cd ~/slurmpilot-code
pip install -e ".[embed]"
```

### Stage 1 — Build the knowledge base

```bash
slurmpilot kb build
```

This indexes `slurmpilot/knowledge/*.md` into Chroma. To keep the markdown in a separate
folder, set `SLURMPILOT_KNOWLEDGE=/path/to/knowledge` first.

### Stage 2 — List your failed jobs

```bash
slurmpilot jobs --state FAILED,TIMEOUT,OUT_OF_MEMORY
```

Or go straight to SLURM for a wider window:

```bash
sacct -u $USER -X --starttime now-30days --format=JobID,JobName,State,End
```

### Stage 3 — Diagnose one job

```bash
slurmpilot diagnose <jobid> --no-llm --direct
```

On NCSA Delta, against a real job of mine that hit its wall-clock limit:

```
$ slurmpilot diagnose 22542799 --no-llm --direct
╭───────── job 22542799  bash  [TIMEOUT]  exit 0:0  12:00:03/12:00:00 ─────────╮
│ Cause - sacct reports TIMEOUT (rule-based, confidence 95%).                   │
│                                                                              │
│ Evidence                                                                     │
│ ▌ sacct reports TIMEOUT  Elapsed 12:00:03 vs Timelimit 12:00:00              │
│                                                                              │
│ Fix  Wall-clock time limit - Fixes                                           │
│  1 Ask for more time: #SBATCH --time=12:00:00. Check partition max with      │
│    sinfo -o "%P %l"; move to a long partition if needed.                      │
│  2 Checkpoint and requeue: save state every N steps, add #SBATCH             │
│    --signal=B:USR1@300 and trap USR1 to save + scontrol requeue $SLURM_JOB_ID.│
│  3 Estimate from the log: a job at 95% when killed needs ~10% more time.      │
│  4 For sweeps, split into array jobs: #SBATCH --array=0-9.                    │
╰───────────────────────────────── rules + kb ─────────────────────────────────╯
```

It read the real `sacct` record, saw elapsed (12:00:03) just past the limit (12:00:00),
classified it as a timeout at **95%** confidence, and pulled the fix notes from the
knowledge base. Read-only, one command, on the login node.

For a failure with a mechanical fix (host out-of-memory), it also prints a suggested sbatch
diff:

```
╭──── job 1001  train_llama_sft  [OUT_OF_MEMORY]  exit 0:125  00:41:12/08:00:00 ────╮
│ Cause - Host memory limit hit, cgroup killed the step (confidence 95%).            │
│ Evidence                                                                           │
│ > slurmstepd: error: Detected 1 oom-kill event(s) in StepId=1001.batch ...         │
│ > MaxRSS 31.8G vs ReqMem 32G (99%)                                                 │
│ Fix                                                                                │
│  1 Raise the request: #SBATCH --mem=64G  ...                                       │
╰────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────── suggested sbatch change ───────────────────────────╮
│ -#SBATCH --mem=32G                                                          │
│ +#SBATCH --mem=64G                                                          │
╰─────────────────────────────────────────────────────────────────────────────╯
```

A **running** or **completed** job returns a one-line status instead; the full analysis
runs only for failures.

### Stage 4 — Fleet analytics

```bash
slurmpilot stats                      # last 30 days
slurmpilot stats --since now-7days --json
```

Aggregates your `sacct` history into a failure rate, wasted GPU-/CPU-hours, and the energy
that waste cost (kWh and approximate CO2), broken down by failure kind and partition. A
failed job still consumes the power it was allocated, so this quantifies it. Cancelled and
preempted jobs are reported separately, not counted as failures. Real output from a 30-day
window on NCSA Delta:

```
╭─────────────────── SLURM usage, now-30days to now ───────────────────╮
│ Jobs                   12                                             │
│ Failed                 2  (17% failure rate)                          │
│ Completed              0                                              │
│ Cancelled / preempted  10                                            │
│ Wasted GPU-hours       24.0                                           │
│ Wasted CPU-hours       384.0                                          │
│ Wasted energy          15.4 kWh  (~5.7 kg CO2)                        │
╰──────────── failed jobs = wasted compute = wasted energy ────────────╯
   Failures by kind: time_limit x2
   By partition: gpuA100x4  22% (2/9)  15.4 kWh
```

The energy figure is a documented estimate (power per GPU times elapsed hours, default
400 W/GPU), tunable with `--gpu-watts`, not a calibrated measurement.

### Stage 5 — Diagnose a whole batch

```bash
slurmpilot watch --once --no-llm --direct     # diagnose every current failure
slurmpilot chat                               # ReAct agent; the model picks the tools
```

### Stage 6 — Add the LLM explanation

```bash
ollama pull qwen2.5:7b                        # or point at any OpenAI-compatible endpoint
slurmpilot diagnose <jobid> --direct          # drop --no-llm to use the model
```

### Stage 7 — Use the MCP server from another client

```json
{
  "mcpServers": { "slurmpilot": { "command": "slurmpilot-mcp" } }
}
```

* * *

## Mechanism

```
fetch_job ─┬─ not found ──────────────────────────────────────▶ END
           ├─ running / completed ──▶ status_only ────────────▶ END
           └─▶ read_logs ▶ classify ▶ retrieve_kb ▶ suggest_patch ▶ diagnose ▶ END
```

- **classify** runs regex signatures over the log tail plus the `sacct` state, with sanity
  checks (MaxRSS vs ReqMem for OOM, Elapsed vs Timelimit). Rules decide the cause; the LLM
  never invents one.
- **retrieve_kb** queries Chroma with the finding and evidence (retrieval-augmented
  generation), boosting chunks whose front matter declares the matching failure kind.
- **suggest_patch** handles only mechanical fixes (bump `--mem` / `--time`, add `--requeue`,
  add NCCL / cache env vars, add a missing env activation). The diff is printed, never
  applied.
- **diagnose** writes the final explanation with the LLM, or a template built from the same
  findings when no model is reachable.

The `diagnose` path is a deterministic workflow, chosen so the root cause is reproducible.
The `chat` command is a true ReAct agent that decides which tools to call. Both run on one
MCP tool server, so the same tools work with any MCP client.

* * *

## Failures Recognized

Runtime failures: `oom`, `cuda_oom`, `cuda_error`, `time_limit`, `nccl`, `import_error`,
`module_load`, `disk_quota`, `permission`, `file_not_found`, `segfault`.

Scheduling / allocation failures: `qos_limit` (QOS or account limits), `allocation_exhausted`
(out of service units), `dependency` (a prerequisite job that never succeeded),
`resources_unavailable` (requested GPUs/features/nodes not available), plus `node_fail`,
`preempted`, and `cancelled`.

Each has a fix document in `src/slurmpilot/knowledge/` following a fixed schema (see that
folder's README), so adding a new failure type is one markdown file plus `kb build`.
Mechanical sbatch patches are generated for the common runtime cases (OOM, CUDA OOM, time
limit, NCCL, missing module, node failure, disk quota).

Scope: SlurmPilot diagnoses **operational** failures (why the job died or would not start:
memory, time, GPU, environment, I/O, scheduling, allocation). It does not read or debug
your application source code.

* * *

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SLURMPILOT_BACKEND` | auto | `real` or `mock` (auto picks `real` when `sacct` exists) |
| `SLURMPILOT_KNOWLEDGE` | bundled | external folder for the fix-note markdown |
| `SLURMPILOT_EMBEDDINGS` | auto | `st` (sentence-transformers) or `hash` |
| `SLURMPILOT_HOME` | `~/.slurmpilot` | where Chroma is cached |
| `SLURMPILOT_NO_LLM` | | set to `1` to never call a model |
| `SLURMPILOT_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `SLURMPILOT_LLM_MODEL` |  `qwen2.5:7b` | model name |
| `SLURMPILOT_LOG_TAIL` | 300 | lines of each log stream read |


* * *

## Safety

SlurmPilot is strictly read-only on the cluster. It calls only `sacct`, `scontrol`,
`sinfo`, and reads log files. It never submits, cancels, requeues, or edits jobs or files,
and it runs on the login node, never on a compute node.

* * *