from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .backends.base import JobInfo
from .classify import STATE_HINTS

FAILURE_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
SOFT_STATES = {"CANCELLED", "PREEMPTED"}

# Rough energy figures so wasted GPU-hours can be read as wasted energy
DEFAULT_GPU_WATTS = 400.0   # an A100-class card under load
DEFAULT_CPU_WATTS = 15.0    # per allocated CPU core, ballpark
GRID_KG_CO2_PER_KWH = 0.37  # US average grid carbon intensity


def _elapsed_hours(elapsed: str) -> float:
    if not elapsed or elapsed in ("", "INVALID"):
        return 0.0
    days = 0
    s = elapsed
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return 0.0
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    else:
        h, m, sec = 0, 0, nums[0]
    return days * 24 + h + m / 60 + sec / 3600


def _gpu_count(gres: str) -> int:
    if not gres:
        return 0
    # matches gres/gpu=4, gpu:4, and gpu:a100:4 (type between name and count)
    m = re.search(r"gpu(?::[a-z0-9_]+)?[:=](\d+)", gres, re.I)
    return int(m.group(1)) if m else 0


def _cpu_count(job: JobInfo) -> int:
    try:
        return int(job.num_cpus) if job.num_cpus else 0
    except ValueError:
        return 0


def base_state(job: JobInfo) -> str:
    return job.state.split()[0] if job.state else "UNKNOWN"


@dataclass
class EnergyModel:
    gpu_watts: float = DEFAULT_GPU_WATTS
    cpu_watts: float = DEFAULT_CPU_WATTS

    def job_kwh(self, job: JobInfo) -> float:
        hours = _elapsed_hours(job.elapsed)
        watts = _gpu_count(job.gres) * self.gpu_watts + _cpu_count(job) * self.cpu_watts
        return watts * hours / 1000.0


@dataclass
class Stats:
    total_jobs: int = 0
    failed_jobs: int = 0
    soft_jobs: int = 0
    completed_jobs: int = 0
    wasted_gpu_hours: float = 0.0
    wasted_cpu_hours: float = 0.0
    wasted_kwh: float = 0.0
    by_failure_kind: Counter = field(default_factory=Counter)
    by_partition: dict = field(default_factory=dict)       # partition -> [failed, total]
    wasted_kwh_by_partition: Counter = field(default_factory=Counter)

    @property
    def failure_rate(self) -> float:
        return self.failed_jobs / self.total_jobs if self.total_jobs else 0.0

    @property
    def wasted_co2_kg(self) -> float:
        return self.wasted_kwh * GRID_KG_CO2_PER_KWH

    def summary(self) -> dict:
        return {
            "total_jobs": self.total_jobs,
            "failed_jobs": self.failed_jobs,
            "failure_rate": round(self.failure_rate, 4),
            "completed_jobs": self.completed_jobs,
            "cancelled_or_preempted": self.soft_jobs,
            "wasted_gpu_hours": round(self.wasted_gpu_hours, 2),
            "wasted_cpu_hours": round(self.wasted_cpu_hours, 2),
            "wasted_kwh": round(self.wasted_kwh, 2),
            "wasted_co2_kg": round(self.wasted_co2_kg, 2),
            "by_failure_kind": dict(self.by_failure_kind.most_common()),
            "failure_rate_by_partition": {
                p: round(f / t, 4) for p, (f, t) in sorted(self.by_partition.items())
            },
            "wasted_kwh_by_partition": {
                p: round(k, 2) for p, k in self.wasted_kwh_by_partition.most_common()
            },
        }


def _failure_kind(job: JobInfo) -> str:
    bs = base_state(job)
    if bs in STATE_HINTS:
        return STATE_HINTS[bs][0]
    if bs == "FAILED":
        # exit code 0:125 / 0:9 usually means an OOM kill even when state says FAILED
        if job.exit_code in ("0:125", "0:9"):
            return "oom"
        return "failed"
    return bs.lower()


def compute_stats(jobs: list[JobInfo], energy: EnergyModel | None = None) -> Stats:
    energy = energy or EnergyModel()
    st = Stats()
    part_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [failed, total]

    for job in jobs:
        bs = base_state(job)
        st.total_jobs += 1
        part = job.partition or "(unknown)"
        part_counts[part][1] += 1

        if bs in FAILURE_STATES:
            st.failed_jobs += 1
            part_counts[part][0] += 1
            st.by_failure_kind[_failure_kind(job)] += 1
            hours = _elapsed_hours(job.elapsed)
            st.wasted_gpu_hours += _gpu_count(job.gres) * hours
            st.wasted_cpu_hours += _cpu_count(job) * hours
            kwh = energy.job_kwh(job)
            st.wasted_kwh += kwh
            st.wasted_kwh_by_partition[part] += kwh
        elif bs in SOFT_STATES:
            st.soft_jobs += 1
        elif bs == "COMPLETED":
            st.completed_jobs += 1

    st.by_partition = {p: v for p, v in part_counts.items()}
    return st