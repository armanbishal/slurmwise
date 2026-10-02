from __future__ import annotations

import difflib
import math
import re

from .backends.base import JobInfo

MEM_RE = re.compile(r"^(#SBATCH\s+--mem(?:-per-cpu)?=)(\d+(?:\.\d+)?)([KMGT]?)\s*$", re.I)
TIME_RE = re.compile(r"^(#SBATCH\s+--time=)(\S+)\s*$", re.I)


def _parse_time(s: str) -> int:
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    else:
        h, m, sec = 0, parts[0], 0
    return days * 86400 + h * 3600 + m * 60 + sec


def _fmt_time(secs: int) -> str:
    days, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    core = f"{h:02d}:{m:02d}:{s:02d}"
    return f"{days}-{core}" if days else core


def _bump_mem(line: str, factor: float) -> str:
    m = MEM_RE.match(line)
    if not m:
        return line
    val = float(m.group(2)) * factor
    unit = m.group(3).upper() or "M"
    return f"{m.group(1)}{math.ceil(val)}{unit}"


def _bump_time(line: str, factor: float) -> str:
    m = TIME_RE.match(line)
    if not m:
        return line
    secs = _parse_time(m.group(2))
    new = math.ceil(secs * factor / 900) * 900
    return f"{m.group(1)}{_fmt_time(new)}"


def _insert_after_sbatch(lines: list[str], new_lines: list[str]) -> list[str]:
    idx = 0
    for i, line in enumerate(lines):
        if line.startswith("#SBATCH"):
            idx = i + 1
    return lines[:idx] + new_lines + lines[idx:]


def suggest_patch(kind: str, job: JobInfo, script: str | None) -> dict | None:
    if not script:
        return None
    lines = script.splitlines()
    new = list(lines)
    note = ""

    if kind == "oom":
        touched = False
        for i, line in enumerate(new):
            if MEM_RE.match(line):
                new[i] = _bump_mem(line, 2.0)
                touched = True
        if not touched:
            new = _insert_after_sbatch(new, ["#SBATCH --mem=64G"])
        note = "Doubled --mem. Also consider fewer DataLoader workers."
    elif kind == "time_limit":
        touched = False
        for i, line in enumerate(new):
            if TIME_RE.match(line):
                new[i] = _bump_time(line, 1.5)
                touched = True
        if not touched:
            new = _insert_after_sbatch(new, ["#SBATCH --time=08:00:00"])
        note = "Raised --time by 50%. If the partition caps below that, switch partitions or checkpoint+requeue."
    elif kind == "cuda_oom":
        if not any("PYTORCH_CUDA_ALLOC_CONF" in line for line in new):
            new = _insert_after_sbatch(new, ["", "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"])
        note = "Added the allocator hint. The real fix is a smaller batch / shorter sequences / gradient checkpointing in the training config."
    elif kind in ("node_fail", "preempted"):
        if not any("--requeue" in line for line in new):
            new = _insert_after_sbatch(new, ["#SBATCH --requeue"])
        note = "Added --requeue so SLURM resubmits on the next node failure."
    elif kind == "nccl":
        adds = []
        if not any("NCCL_SOCKET_IFNAME" in line for line in new):
            adds.append("export NCCL_SOCKET_IFNAME=ib0   # check `ip link` on a node")
        if not any("NCCL_DEBUG" in line for line in new):
            adds.append("export NCCL_DEBUG=INFO")
        if adds:
            new = _insert_after_sbatch(new, [""] + adds)
        note = "Pinned NCCL to the IB interface and turned on debug logging for the retry."
    elif kind == "disk_quota":
        if not any("HF_HOME" in line for line in new):
            new = _insert_after_sbatch(
                new,
                ["", "export HF_HOME=/scratch/$USER/hf", "export TORCH_HOME=/scratch/$USER/torch", "export PIP_CACHE_DIR=/scratch/$USER/pip"],
            )
        note = "Pointed caches at scratch. Adjust the scratch path for your cluster."
    elif kind == "import_error":
        has_activate = any(re.search(r"(source .*activate|conda activate|uv run|poetry run)", line) for line in new)
        if not has_activate:
            new = _insert_after_sbatch(new, ["", "source ~/.venv/bin/activate   # <- the job never activated an env", "which python"])
            note = "The script never activates a Python env. Added an activation line (fix the path)."
        else:
            note = "Env is activated; the package is missing from that env. Install it from a compute node."
            return {"diff": "", "note": note}
    else:
        return None

    if new == lines:
        return None
    diff = "\n".join(
        difflib.unified_diff(lines, new, fromfile=f"job{job.job_id}.sbatch", tofile=f"job{job.job_id}.sbatch (suggested)", lineterm="")
    )
    return {"diff": diff, "note": note}