from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

# States where the job is over. Anything else we treat as "still going".
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


# everything we know about one job, filled from sacct + scontrol
@dataclass
class JobInfo:
    job_id: str
    name: str = ""
    user: str = ""
    partition: str = ""
    state: str = "UNKNOWN"
    exit_code: str = ""
    elapsed: str = ""
    time_limit: str = ""
    req_mem: str = ""
    max_rss: str = ""
    nodes: str = ""
    num_cpus: str = ""
    gres: str = ""
    submit_time: str = ""
    start_time: str = ""
    end_time: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    work_dir: str = ""
    reason: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        # sacct sometimes reports "CANCELLED by 12345"
        return self.state.split()[0] in TERMINAL_STATES

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_terminal"] = self.is_terminal
        return d

class SlurmBackend(Protocol):
    def get_job(self, job_id: str) -> JobInfo: ...

    def list_jobs(self, user: str | None = None, states: list[str] | None = None, limit: int = 20) -> list[JobInfo]: ...

    def history(self, user: str | None = None, since: str = "now-30days") -> list[JobInfo]: ...

    def read_log(self, job_id: str, which: str = "stderr", tail: int = 200) -> str: ...

    def get_script(self, job_id: str) -> str | None: ...

    def cluster_info(self) -> str: ...


# keep only the last n lines (logs can be huge, the error is at the end)
def tail_text(text: str, n: int) -> str:
    if n <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])
