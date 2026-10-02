import pytest

from slurmwise.classify import classify

CASES = {
    "1001": "oom",
    "1002": "time_limit",
    "1003": "cuda_oom",
    "1004": "cuda_error",
    "1005": "import_error",
    "1008": "nccl",
    "1009": "node_fail",
    "1010": "disk_quota",
    "1011": "cancelled",
}


@pytest.mark.parametrize("job_id,kind", CASES.items())
def test_top_finding(backend, job_id, kind):
    job = backend.get_job(job_id)
    f = classify(job, backend.read_log(job_id, "stderr"), backend.read_log(job_id, "stdout"))
    assert f[0].kind == kind, [x.kind for x in f]
    assert f[0].evidence


def test_completed_and_running(backend):
    assert classify(backend.get_job("1006"), "", "")[0].kind == "ok"
    assert classify(backend.get_job("1007"), "", "")[0].kind == "running"


def test_cuda_oom_demotes_generic_cuda_error(backend):
    text = "RuntimeError: CUDA error: out of memory\ntorch.OutOfMemoryError: CUDA out of memory."
    f = classify(backend.get_job("1003"), text, "")
    kinds = {x.kind: x.confidence for x in f}
    assert kinds["cuda_oom"] > kinds["cuda_error"]


def test_state_only_no_logs(backend):
    f = classify(backend.get_job("1002"), "", "")
    assert f[0].kind == "time_limit"
    assert any("Elapsed" in e for e in f[0].evidence)


def test_oom_mem_ratio_evidence(backend):
    f = classify(backend.get_job("1001"), backend.read_log("1001"), "")
    assert any("MaxRSS" in e for e in f[0].evidence)


def test_unknown():
    f = classify(None, "something odd happened\n", "")
    assert f[0].kind == "unknown"
