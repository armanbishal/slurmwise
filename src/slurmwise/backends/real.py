from __future__ import annotations

import os
import subprocess

from .base import JobInfo, tail_text

SACCT_FIELDS = [
    "JobID",
    "JobName",
    "User",
    "Partition",
    "State",
    "ExitCode",
    "Elapsed",
    "Timelimit",
    "ReqMem",
    "MaxRSS",
    "NodeList",
    "AllocCPUS",
    "ReqTRES",
    "Submit",
    "Start",
    "End",
    "WorkDir",
    "Reason",
]


# run a read-only SLURM command and return its stdout
def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} not found on PATH; is this a SLURM login node?") from e
    if out.returncode != 0 and not out.stdout.strip():
        raise RuntimeError(f"{' '.join(cmd)} failed: {out.stderr.strip()}")
    return out.stdout


def _parse_scontrol(text: str) -> dict:
    kv: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # keys that own the rest of the line
        for solo in ("Command=", "WorkDir=", "StdErr=", "StdOut=", "StdIn=", "Comment="):
            if line.startswith(solo):
                kv[solo[:-1]] = line[len(solo):]
                break
        else:
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
    return kv


# Talk to a live cluster via sacct/scontrol/sinfo, all read-only
class RealBackend:
    def _sacct_rows(self, args: list[str]) -> list[dict]:
        text = _run(["sacct", "-P", "-n", "--format=" + ",".join(SACCT_FIELDS), *args])
        rows = []
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) < len(SACCT_FIELDS):
                continue
            rows.append(dict(zip(SACCT_FIELDS, parts)))
        return rows

    # sacct returns the job plus .batch/.extern steps; fold MaxRSS back into the parent
    def _merge_steps(self, rows: list[dict]) -> dict | None:
        parent = None
        max_rss = ""
        for r in rows:
            if "." not in r["JobID"]:
                parent = r
            elif r["MaxRSS"]:
                max_rss = max([max_rss, r["MaxRSS"]], key=_mem_to_bytes)
        if parent is not None and not parent.get("MaxRSS"):
            parent["MaxRSS"] = max_rss
        return parent

    def get_job(self, job_id: str) -> JobInfo:
        rows = self._sacct_rows(["-j", job_id])
        parent = self._merge_steps(rows)
        if parent is None:
            raise KeyError(f"job {job_id} not found in sacct")
        job = JobInfo(
            job_id=parent["JobID"],
            name=parent["JobName"],
            user=parent["User"],
            partition=parent["Partition"],
            state=parent["State"],
            exit_code=parent["ExitCode"],
            elapsed=parent["Elapsed"],
            time_limit=parent["Timelimit"],
            req_mem=parent["ReqMem"],
            max_rss=parent["MaxRSS"],
            nodes=parent["NodeList"],
            num_cpus=parent["AllocCPUS"],
            gres=parent["ReqTRES"],
            submit_time=parent["Submit"],
            start_time=parent["Start"],
            end_time=parent["End"],
            work_dir=parent["WorkDir"],
            reason=parent["Reason"],
        )
        try:
            kv = _parse_scontrol(_run(["scontrol", "show", "job", job_id]))
            job.stdout_path = kv.get("StdOut", "")
            job.stderr_path = kv.get("StdErr", "")
            job.extra["command"] = kv.get("Command", "")
            if not job.work_dir:
                job.work_dir = kv.get("WorkDir", "")
        except RuntimeError:
            pass
        if not job.stdout_path:
            job.stdout_path = self._guess_log(job)
        if not job.stderr_path:
            job.stderr_path = job.stdout_path
        return job

    @staticmethod
    def _guess_log(job: JobInfo) -> str:
        cand = os.path.join(job.work_dir or ".", f"slurm-{job.job_id}.out")
        return cand if os.path.exists(cand) else ""

    def list_jobs(self, user: str | None = None, states: list[str] | None = None, limit: int = 20) -> list[JobInfo]:
        args = ["-S", "now-7days", "-X"]
        args += ["-u", user] if user else ["-u", os.environ.get("USER", "")]
        if states:
            args += ["-s", ",".join(states)]
        rows = self._sacct_rows(args)
        rows.sort(key=lambda r: r["Submit"], reverse=True)
        return [
            JobInfo(
                job_id=r["JobID"], name=r["JobName"], user=r["User"], partition=r["Partition"],
                state=r["State"], exit_code=r["ExitCode"], elapsed=r["Elapsed"], time_limit=r["Timelimit"],
                req_mem=r["ReqMem"], nodes=r["NodeList"], submit_time=r["Submit"], end_time=r["End"],
                work_dir=r["WorkDir"],
            )
            for r in rows[:limit]
        ]

    def history(self, user: str | None = None, since: str = "now-30days") -> list[JobInfo]:
        args = ["-S", since, "-X", "-u", user or os.environ.get("USER", "")]
        rows = self._sacct_rows(args)
        return [
            JobInfo(
                job_id=r["JobID"], name=r["JobName"], user=r["User"], partition=r["Partition"],
                state=r["State"], exit_code=r["ExitCode"], elapsed=r["Elapsed"], time_limit=r["Timelimit"],
                req_mem=r["ReqMem"], max_rss=r["MaxRSS"], nodes=r["NodeList"], num_cpus=r["AllocCPUS"],
                gres=r["ReqTRES"], submit_time=r["Submit"], end_time=r["End"], work_dir=r["WorkDir"],
            )
            for r in rows
        ]

    def read_log(self, job_id: str, which: str = "stderr", tail: int = 200) -> str:
        job = self.get_job(job_id)
        path = job.stderr_path if which == "stderr" else job.stdout_path
        path = path.replace("%j", job.job_id).replace("%J", job.job_id)
        if not path or not os.path.exists(path):
            return f"<no {which} file found for job {job_id} (looked at {path or 'nothing'})>"
        with open(path, errors="replace") as f:
            return tail_text(f.read(), tail)

    def get_script(self, job_id: str) -> str | None:
        try:
            text = _run(["scontrol", "write", "batch_script", job_id, "-"])
            return text if text.strip() else None
        except RuntimeError:
            pass
        job = self.get_job(job_id)
        cmd = job.extra.get("command", "").split()[0] if job.extra.get("command") else ""
        if cmd and os.path.exists(cmd):
            with open(cmd, errors="replace") as f:
                return f.read()
        return None

    def cluster_info(self) -> str:
        return _run(["sinfo", "-o", "%P %a %l %D %T %G"])


def _mem_to_bytes(s: str) -> float:
    s = s.strip().upper()
    if not s:
        return 0.0
    mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    unit = s[-1]
    if unit in mult:
        try:
            return float(s[:-1]) * mult[unit]
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0