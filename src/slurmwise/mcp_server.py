from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .backends import get_backend
from .classify import classify

server = MCPServer(
    name="slurmpilot",
    instructions=(
        "Read-only access to a SLURM cluster: job status (sacct/scontrol), job logs, "
        "batch scripts, partition info, and a rule-based failure classifier. "
        "Nothing here submits or cancels jobs."
    ),
)

_backend = None


def backend():
    global _backend
    if _backend is None:
        _backend = get_backend()
    return _backend


@server.tool(name="get_job", description="Get sacct/scontrol details for one job id: state, exit code, elapsed vs time limit, memory requested vs used, nodes, log paths.")
def get_job(job_id: str) -> str:
    try:
        return json.dumps(backend().get_job(str(job_id)).to_dict(), indent=1)
    except KeyError as e:
        return json.dumps({"error": str(e)})


@server.tool(name="list_jobs", description="List recent jobs for a user (default: current user). Optional state filter like FAILED,TIMEOUT,OUT_OF_MEMORY.")
def list_jobs(user: str | None = None, states: str | None = None, limit: int = 20) -> str:
    st = [s.strip() for s in states.split(",")] if states else None
    jobs = backend().list_jobs(user=user, states=st, limit=limit)
    return json.dumps([j.to_dict() for j in jobs], indent=1)


@server.tool(name="read_log", description="Read the tail of a job's stdout or stderr. `which` is 'stderr' (default) or 'stdout'. `tail` is number of lines.")
def read_log(job_id: str, which: str = "stderr", tail: int = 200) -> str:
    return backend().read_log(str(job_id), which=which, tail=tail)


@server.tool(name="get_script", description="Return the sbatch script that launched the job, if SLURM still has it.")
def get_script(job_id: str) -> str:
    return backend().get_script(str(job_id)) or f"<script for job {job_id} not available>"


@server.tool(name="cluster_info", description="Partition summary from sinfo: names, time limits, node states, GPUs.")
def cluster_info() -> str:
    return backend().cluster_info()


@server.tool(name="classify_failure", description="Run the rule-based failure classifier on a job (reads its logs). Returns ranked findings with evidence lines. Kinds: oom, cuda_oom, time_limit, cuda_error, nccl, import_error, node_fail, disk_quota, ...")
def classify_failure(job_id: str, tail: int = 300) -> str:
    b = backend()
    job = b.get_job(str(job_id))
    err = b.read_log(job.job_id, "stderr", tail)
    out = b.read_log(job.job_id, "stdout", tail)
    return json.dumps([f.to_dict() for f in classify(job, err, out)], indent=1)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()