from __future__ import annotations

import re
from dataclasses import dataclass, field

from .backends.base import JobInfo


@dataclass
class Finding:
    kind: str
    confidence: float
    summary: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "confidence": self.confidence, "summary": self.summary, "evidence": self.evidence}


SIGNATURES: list[tuple[str, float, str, list[str]]] = [
    ("cuda_oom", 0.95, "GPU ran out of memory",
     [r"CUDA out of memory", r"torch\.OutOfMemoryError", r"CUDA_ERROR_OUT_OF_MEMORY", r"cudaErrorMemoryAllocation"]),
    ("oom", 0.95, "Host memory limit hit, cgroup killed the step",
     [r"oom-kill event", r"Out Of Memory", r"OUT_OF_MEMORY", r"Killed process \d+", r"MemoryError$", r"DUE TO PREEMPTION OOM"]),
    ("time_limit", 0.98, "Job hit its wall-clock limit",
     [r"DUE TO TIME LIMIT", r"CANCELLED .* TIME LIMIT"]),
    ("node_fail", 0.95, "Compute node failed under the job",
     [r"DUE TO NODE FAILURE", r"NODE_FAIL"]),
    ("nccl", 0.9, "NCCL / distributed communication error",
     [r"NCCL error", r"ncclSystemError", r"ncclUnhandledCudaError", r"NCCL WARN", r"DistBackendError", r"Connection refused.*29\d{3}"]),
    ("cuda_error", 0.9, "CUDA runtime / kernel error",
     [r"CUDA error: device-side assert", r"device-side assert triggered", r"CUDA error:", r"no kernel image is available",
      r"CUDA driver version is insufficient", r"CUBLAS_STATUS_", r"cuDNN error", r"RuntimeError: CUDA"]),
    ("import_error", 0.95, "Python module missing in the job environment",
     [r"ModuleNotFoundError: No module named", r"ImportError: cannot import name", r"ImportError: .*\.so"]),
    ("disk_quota", 0.95, "Disk quota / no space",
     [r"Disk quota exceeded", r"No space left on device", r"\[Errno 122\]", r"\[Errno 28\]"]),
    ("permission", 0.85, "Permission problem on a file or directory",
     [r"PermissionError", r"\[Errno 13\]", r"Permission denied"]),
    ("file_not_found", 0.85, "File or directory missing",
     [r"FileNotFoundError", r"\[Errno 2\] No such file", r"No such file or directory"]),
    ("module_load", 0.85, "Environment module could not be loaded",
     [r"Lmod has detected the following error", r"module: command not found", r"Unable to locate a modulefile"]),
    ("qos_limit", 0.9, "Blocked by a QOS or account limit",
     [r"QOSMax\w+Limit", r"AssocMax\w+Limit", r"QOSMaxJobsPerUserLimit", r"Invalid\s*QOS", r"Invalid\s*account",
      r"Invalid qos specification", r"Job violates accounting/QOS policy", r"AssocGrp\w+Limit"]),
    ("allocation_exhausted", 0.9, "Allocation / billing balance exhausted",
     [r"AssocGrpBillingMinutes", r"AssocGrpCPUMinutesLimit", r"out of .*service units", r"insufficient.*(allocation|balance|SUs)",
      r"account has no .*remaining"]),
    ("dependency", 0.9, "A job dependency was never satisfied",
     [r"DependencyNeverSatisfied", r"Dependency\b.*never", r"\(Dependency\)"]),
    ("resources_unavailable", 0.75, "Requested resources never became available",
     [r"ReqNodeNotAvail", r"Nodes required for job are DOWN", r"PartitionNodeLimit", r"Requested node configuration is not available"]),
    ("cancelled", 0.9, "Job was cancelled (by a user or admin)",
     [r"CANCELLED AT .* \*\*\*$"]),
    ("python_exception", 0.6, "Python raised an uncaught exception",
     [r"^Traceback \(most recent call last\)", r"^\w+(Error|Exception): "]),
    ("segfault", 0.9, "Process crashed with a signal",
     [r"Segmentation fault", r"core dumped", r"Bus error", r"Illegal instruction"]),
]

STATE_HINTS = {
    "OUT_OF_MEMORY": ("oom", 0.9, "sacct reports OUT_OF_MEMORY"),
    "TIMEOUT": ("time_limit", 0.95, "sacct reports TIMEOUT"),
    "NODE_FAIL": ("node_fail", 0.95, "sacct reports NODE_FAIL"),
    "CANCELLED": ("cancelled", 0.7, "sacct reports CANCELLED"),
    "PREEMPTED": ("preempted", 0.95, "sacct reports PREEMPTED"),
}

def _grep(text: str, patterns: list[str], max_hits: int = 4) -> list[str]:
    hits: list[str] = []
    for line in text.splitlines():
        for p in patterns:
            if re.search(p, line):
                s = line.strip()
                if s and s not in hits:
                    hits.append(s[:300])
                break
        if len(hits) >= max_hits:
            break
    return hits


def classify(job: JobInfo | None, stderr: str = "", stdout: str = "") -> list[Finding]:
    text = (stderr or "") + "\n" + (stdout or "")
    findings: dict[str, Finding] = {}

    for kind, conf, summary, pats in SIGNATURES:
        hits = _grep(text, pats)
        if hits:
            findings[kind] = Finding(kind, conf, summary, hits)

    if job is not None:
        base_state = job.state.split()[0]
        if base_state in STATE_HINTS:
            kind, conf, why = STATE_HINTS[base_state]
            if kind in findings:
                findings[kind].confidence = max(findings[kind].confidence, conf)
                findings[kind].evidence.insert(0, why)
            else:
                findings[kind] = Finding(kind, conf, STATE_HINTS[base_state][2], [why])

        if "cancelled" in findings and any(k in findings for k in ("oom", "time_limit", "node_fail")):
            del findings["cancelled"]

        if "python_exception" in findings and len(findings) > 1:
            findings["python_exception"].confidence = 0.3

        if "cuda_oom" in findings and "cuda_error" in findings:
            findings["cuda_error"].confidence = 0.3

        if "oom" in findings and job.max_rss and job.req_mem:
            r, m = _to_gb(job.max_rss), _to_gb(job.req_mem)
            if r and m and r / m > 0.9:
                findings["oom"].evidence.append(f"MaxRSS {job.max_rss} vs ReqMem {job.req_mem} ({r / m:.0%})")

        if "time_limit" in findings and job.elapsed and job.time_limit:
            findings["time_limit"].evidence.append(f"Elapsed {job.elapsed} vs Timelimit {job.time_limit}")

    if not findings:
        if job is not None and job.state.split()[0] == "COMPLETED":
            return [Finding("ok", 1.0, "Job completed with exit code %s" % (job.exit_code or "0:0"), [])]
        if job is not None and not job.is_terminal:
            return [Finding("running", 1.0, f"Job is {job.state}, nothing to diagnose yet", [])]
        return [Finding("unknown", 0.2, "No known failure signature in the logs", [])]

    return sorted(findings.values(), key=lambda f: -f.confidence)


# parse a SLURM memory string into gigabytes
def _to_gb(s: str) -> float | None:
    m = re.match(r"([\d.]+)\s*([KMGT]?)", s.strip().upper())
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    return val * {"": 1 / 1024**3, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1, "T": 1024}[unit]
