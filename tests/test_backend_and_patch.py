from slurmwise.backends.base import JobInfo
from slurmwise.backends.real import _parse_scontrol
from slurmwise.patch import _fmt_time, _parse_time, suggest_patch


def test_mock_list_and_filter(backend):
    failed = backend.list_jobs(user="arman", states=["FAILED"])
    assert {j.job_id for j in failed} == {"1003", "1004", "1005", "1008", "1010"}
    assert backend.list_jobs(limit=3).__len__() == 3


def test_mock_log_tail(backend):
    assert backend.read_log("1002", "stdout", tail=1).strip().endswith("DUE TO TIME LIMIT ***")
    assert "no stderr" not in backend.read_log("1002", "stderr")


def test_is_terminal():
    assert JobInfo("1", state="CANCELLED by 5012").is_terminal
    assert not JobInfo("1", state="PENDING").is_terminal


def test_time_roundtrip():
    for s in ["04:00:00", "1-12:00:00", "30:00", "45"]:
        assert _parse_time(_fmt_time(_parse_time(s))) == _parse_time(s)


def test_patch_oom(backend):
    p = suggest_patch("oom", backend.get_job("1001"), backend.get_script("1001"))
    assert "+#SBATCH --mem=64G" in p["diff"]
    assert "-#SBATCH --mem=32G" in p["diff"]


def test_patch_time(backend):
    p = suggest_patch("time_limit", backend.get_job("1002"), backend.get_script("1002"))
    assert "+#SBATCH --time=06:00:00" in p["diff"]


def test_patch_import_error_adds_activate(backend):
    p = suggest_patch("import_error", backend.get_job("1005"), backend.get_script("1005"))
    assert "source" in p["diff"]
    p2 = suggest_patch("import_error", backend.get_job("1003"), backend.get_script("1003"))
    assert p2["diff"] == "" and "missing" in p2["note"]


def test_patch_none_without_script(backend):
    assert suggest_patch("oom", backend.get_job("1004"), None) is None


def test_scontrol_parse():
    text = "JobId=12 JobName=x\n   UserId=arman(1) GroupId=g(2)\n   Command=/home/a/run.sh --x 1\n   StdErr=/home/a/logs/%j.err\n"
    kv = _parse_scontrol(text)
    assert kv["JobId"] == "12" and kv["Command"] == "/home/a/run.sh --x 1" and kv["StdErr"].endswith("%j.err")
