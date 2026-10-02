from __future__ import annotations

import json
import os
from pathlib import Path

from .base import JobInfo, tail_text

_HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURES = _HERE.parent.parent.parent / "tests" / "fixtures"


class MockBackend:
    def __init__(self, fixtures_dir: str | os.PathLike | None = None):
        self.root = Path(fixtures_dir) if fixtures_dir else DEFAULT_FIXTURES
        with open(self.root / "jobs.json") as f:
            raw = json.load(f)
        self.jobs: dict[str, JobInfo] = {}
        for d in raw:
            job = JobInfo(**d)
            job.stdout_path = job.stdout_path or str(self.root / "logs" / f"{job.job_id}.out")
            job.stderr_path = job.stderr_path or str(self.root / "logs" / f"{job.job_id}.err")
            self.jobs[job.job_id] = job

    def get_job(self, job_id: str) -> JobInfo:
        try:
            return self.jobs[str(job_id)]
        except KeyError:
            raise KeyError(f"job {job_id} not found") from None

    def list_jobs(self, user: str | None = None, states: list[str] | None = None, limit: int = 20) -> list[JobInfo]:
        out = list(self.jobs.values())
        if user:
            out = [j for j in out if j.user == user]
        if states:
            wanted = {s.upper() for s in states}
            out = [j for j in out if j.state.split()[0] in wanted]
        out.sort(key=lambda j: j.submit_time, reverse=True)
        return out[:limit]

    def history(self, user: str | None = None, since: str = "now-30days") -> list[JobInfo]:
        # fixtures have no real timestamps to window on; return everything (optionally by user)
        jobs = list(self.jobs.values())
        return [j for j in jobs if not user or j.user == user]

    def read_log(self, job_id: str, which: str = "stderr", tail: int = 200) -> str:
        job = self.get_job(job_id)
        path = Path(job.stderr_path if which == "stderr" else job.stdout_path)
        if not path.exists() and which == "stderr":
            path = Path(job.stdout_path)  # sbatch merges streams unless -e given
        if not path.exists():
            return f"<no {which} file found for job {job_id}>"
        return tail_text(path.read_text(errors="replace"), tail)

    def get_script(self, job_id: str) -> str | None:
        p = self.root / "scripts" / f"{job_id}.sh"
        return p.read_text() if p.exists() else None

    def cluster_info(self) -> str:
        p = self.root / "sinfo.txt"
        return p.read_text() if p.exists() else "PARTITION AVAIL TIMELIMIT NODES STATE GRES\n(no fixture)"
