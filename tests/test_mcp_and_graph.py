import json
import sys

import pytest

from slurmwise.agent.graph import build_graph, diagnose_job
from slurmwise.agent.tools import DirectToolClient, MCPToolClient, as_langchain_tools


def test_tools_listed(backend):
    c = DirectToolClient(backend)
    names = {t["name"] for t in c.list_tools()}
    assert names == {"get_job", "list_jobs", "read_log", "get_script", "cluster_info", "classify_failure"}


def test_direct_tool_calls(backend):
    c = DirectToolClient(backend)
    job = json.loads(c.call("get_job", job_id="1001"))
    assert job["state"] == "OUT_OF_MEMORY"
    assert json.loads(c.call("get_job", job_id="nope"))["error"]
    findings = json.loads(c.call("classify_failure", job_id="1002"))
    assert findings[0]["kind"] == "time_limit"
    assert "gpu-long" in c.call("cluster_info")


def test_graph_full_path(backend, kb):
    r = diagnose_job("1003", client=DirectToolClient(backend), kb=kb, use_llm=False)
    assert r["findings"][0]["kind"] == "cuda_oom"
    assert r["kb_hits"] and r["kb_hits"][0]["title"] == "GPU out of memory"
    assert "PYTORCH_CUDA_ALLOC_CONF" in r["patch"]["diff"]
    assert r["used_llm"] is False
    assert "**Cause**" in r["diagnosis"]


def test_graph_short_circuits(backend, kb):
    running = diagnose_job("1007", client=DirectToolClient(backend), kb=kb, use_llm=False)
    assert "RUNNING" in running["diagnosis"] and not running["findings"]
    done = diagnose_job("1006", client=DirectToolClient(backend), kb=kb, use_llm=False)
    assert "Nothing to fix" in done["diagnosis"]
    missing = diagnose_job("4242", client=DirectToolClient(backend), kb=kb, use_llm=False)
    assert missing["error"] and "diagnosis" not in missing


def test_graph_llm_failure_falls_back(backend, kb, monkeypatch):
    monkeypatch.delenv("SLURMPILOT_NO_LLM", raising=False)
    monkeypatch.setenv("SLURMPILOT_LLM_BASE_URL", "http://127.0.0.1:9")
    g = build_graph(DirectToolClient(backend), kb, use_llm=True)
    r = g.invoke({"job_id": "1001"})
    assert r["used_llm"] is False and "LLM unavailable" in r["diagnosis"]



def test_mcp_stdio_roundtrip(kb):
    with MCPToolClient(sys.executable, ["-m", "slurmwise.mcp_server"]) as c:
        names = {t["name"] for t in c.list_tools()}
        assert "classify_failure" in names
        out = json.loads(c.call("classify_failure", job_id="1008"))
        assert out[0]["kind"] == "nccl"
        r = diagnose_job("1009", client=c, kb=kb, use_llm=False)
        assert r["findings"][0]["kind"] == "node_fail" and "--requeue" in r["patch"]["diff"]


def test_langchain_tool_wrappers(backend):
    tools = as_langchain_tools(DirectToolClient(backend))
    t = {x.name: x for x in tools}["read_log"]
    out = t.invoke({"job_id": "1010", "which": "stderr", "tail": 5})
    assert "Disk quota exceeded" in out
