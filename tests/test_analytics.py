from slurmwise.analytics import (EnergyModel, _cpu_count, _elapsed_hours, _gpu_count,
                                  base_state, compute_stats)
from slurmwise.backends.base import JobInfo


def test_elapsed_hours():
    assert _elapsed_hours("01:00:00") == 1.0
    assert _elapsed_hours("00:30:00") == 0.5
    assert abs(_elapsed_hours("1-12:00:00") - 36.0) < 1e-9
    assert _elapsed_hours("45:00") == 0.75          # MM:SS
    assert _elapsed_hours("") == 0.0


def test_gpu_and_cpu_count():
    assert _gpu_count("billing=8,cpu=8,gres/gpu=2,mem=32G") == 2
    assert _gpu_count("gpu:4") == 4
    assert _gpu_count("cpu=4,mem=16G") == 0
    assert _cpu_count(JobInfo("1", num_cpus="8")) == 8
    assert _cpu_count(JobInfo("1", num_cpus="")) == 0


def test_compute_stats_basic(ds_jobs):
    st = compute_stats(ds_jobs)
    assert st.total_jobs == len(ds_jobs)
    assert st.failed_jobs >= 1
    assert 0 <= st.failure_rate <= 1
    kinds = st.by_failure_kind
    assert kinds["oom"] >= 1
    assert kinds["time_limit"] >= 1
    assert kinds["node_fail"] >= 1


def test_wasted_energy_only_counts_failures():
    good = JobInfo("1", state="COMPLETED", elapsed="10:00:00", gres="gres/gpu=1", num_cpus="8")
    bad = JobInfo("2", state="FAILED", elapsed="10:00:00", gres="gres/gpu=1", num_cpus="8")
    st = compute_stats([good, bad])
    assert st.wasted_gpu_hours == 10.0
    assert st.wasted_cpu_hours == 80.0
    assert st.wasted_kwh > 0


def test_energy_model_scales_with_watts():
    job = JobInfo("1", state="FAILED", elapsed="01:00:00", gres="gres/gpu=1", num_cpus="0")
    low = compute_stats([job], EnergyModel(gpu_watts=100)).wasted_kwh
    high = compute_stats([job], EnergyModel(gpu_watts=400)).wasted_kwh
    assert abs(high - 4 * low) < 1e-6
    assert abs(low - 0.1) < 1e-6


def test_partition_breakdown():
    jobs = [
        JobInfo("1", state="FAILED", partition="gpu", elapsed="01:00:00", gres="gres/gpu=1"),
        JobInfo("2", state="COMPLETED", partition="gpu", elapsed="01:00:00", gres="gres/gpu=1"),
        JobInfo("3", state="FAILED", partition="cpu", elapsed="01:00:00", num_cpus="4"),
    ]
    st = compute_stats(jobs)
    assert st.by_partition["gpu"] == [1, 2]
    assert st.by_partition["cpu"] == [1, 1]
    s = st.summary()
    assert s["failure_rate_by_partition"]["gpu"] == 0.5


def test_empty_input():
    st = compute_stats([])
    assert st.total_jobs == 0 and st.failure_rate == 0.0


def test_base_state_strips_suffix():
    assert base_state(JobInfo("1", state="CANCELLED by 5012")) == "CANCELLED"


def test_backend_history(backend):
    jobs = backend.history()
    assert len(jobs) >= 10
    assert all(isinstance(j, JobInfo) for j in jobs)