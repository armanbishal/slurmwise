from __future__ import annotations

import json
import os

from langgraph.graph import END, StateGraph

from ..backends.base import JobInfo
from ..kb import KnowledgeBase
from ..patch import suggest_patch as _suggest_patch
from .llm import get_llm
from .prompts import SYSTEM, diagnosis_prompt
from .state import DiagnosisState
from .tools import ToolClient, get_tool_client, loads_or_text

LOG_TAIL = int(os.environ.get("SLURMPILOT_LOG_TAIL", "300"))


def build_graph(client: ToolClient, kb: KnowledgeBase | None = None, use_llm: bool = True):
    llm = get_llm(use_llm)

    # Pull the sacct/scontrol record for this job
    def fetch_job(state: DiagnosisState) -> DiagnosisState:
        raw = loads_or_text(client.call("get_job", job_id=state["job_id"]))
        if not isinstance(raw, dict) or "error" in raw:
            return {"job": None, "error": raw.get("error") if isinstance(raw, dict) else str(raw)}
        return {"job": raw, "error": None}

    # Stop early if the job is missing, running, or finished fine
    def route_after_fetch(state: DiagnosisState) -> str:
        job = state.get("job")
        if job is None:
            return "end"
        base = job["state"].split()[0]
        if not job.get("is_terminal") or base == "COMPLETED":
            return "status_only"
        return "read_logs"

    # Report stats
    def status_only(state: DiagnosisState) -> DiagnosisState:
        job = state["job"]
        base = job["state"].split()[0]
        if base == "COMPLETED":
            msg = f"Job {job['job_id']} ({job['name']}) completed in {job['elapsed']} with exit code {job['exit_code']}. Nothing to fix."
        else:
            msg = f"Job {job['job_id']} ({job['name']}) is {job['state']} on {job['nodes'] or 'no node yet'}, {job['elapsed']} of {job['time_limit']} used."
            if job.get("reason") and job["reason"] != "None":
                msg += f" Reason: {job['reason']}."
        return {"diagnosis": msg, "findings": [], "kb_hits": [], "patch": None, "used_llm": False}

    # Read the tail of stderr/stdout and the sbatch script
    def read_logs(state: DiagnosisState) -> DiagnosisState:
        jid = state["job_id"]
        return {
            "stderr": client.call("read_log", job_id=jid, which="stderr", tail=LOG_TAIL),
            "stdout": client.call("read_log", job_id=jid, which="stdout", tail=LOG_TAIL),
            "script": _none_if_missing(client.call("get_script", job_id=jid)),
        }

    # Rules decide the cause from the logs + sacct state
    def classify_node(state: DiagnosisState) -> DiagnosisState:
        raw = loads_or_text(client.call("classify_failure", job_id=state["job_id"], tail=LOG_TAIL))
        return {"findings": raw if isinstance(raw, list) else []}

    # Pull matching fix notes from the Chroma knowledge base (RAG)
    def retrieve_kb(state: DiagnosisState) -> DiagnosisState:
        if kb is None:
            return {"kb_hits": []}
        findings = state.get("findings") or []
        kinds = [f["kind"] for f in findings[:2]]
        query = " ".join([f["summary"] for f in findings[:2]] + [e for f in findings[:1] for e in f["evidence"][:2]])
        hits = kb.query(query or "slurm job failed", kinds=kinds, k=4)
        return {"kb_hits": [h.to_dict() for h in hits]}

    # Bild an sbatch diff for fixes
    def suggest_patch(state: DiagnosisState) -> DiagnosisState:
        findings = state.get("findings") or []
        if not findings:
            return {"patch": None}
        job = JobInfo(**{k: v for k, v in state["job"].items() if k != "is_terminal"})
        return {"patch": _suggest_patch(findings[0]["kind"], job, state.get("script"))}

    def diagnose(state: DiagnosisState) -> DiagnosisState:
        findings = state.get("findings") or []
        prompt = diagnosis_prompt(
            state["job"], findings, state.get("kb_hits") or [], state.get("stderr", ""), state.get("stdout", ""),
            state.get("script"), state.get("patch"),
        )
        if llm is not None:
            try:
                resp = llm.invoke([("system", SYSTEM), ("user", prompt)])
                return {"diagnosis": resp.content, "used_llm": True}
            except Exception as e:  # endpoint down, model missing, whatever: fall back
                fallback = _template_diagnosis(state) + f"\n\n(LLM unavailable: {type(e).__name__}: {str(e)[:120]})"
                return {"diagnosis": fallback, "used_llm": False}
        return {"diagnosis": _template_diagnosis(state), "used_llm": False}

    g = StateGraph(DiagnosisState)
    g.add_node("fetch_job", fetch_job)
    g.add_node("status_only", status_only)
    g.add_node("read_logs", read_logs)
    g.add_node("classify", classify_node)
    g.add_node("retrieve_kb", retrieve_kb)
    g.add_node("suggest_patch", suggest_patch)
    g.add_node("diagnose", diagnose)
    g.set_entry_point("fetch_job")
    g.add_conditional_edges("fetch_job", route_after_fetch, {"end": END, "status_only": "status_only", "read_logs": "read_logs"})
    g.add_edge("status_only", END)
    g.add_edge("read_logs", "classify")
    g.add_edge("classify", "retrieve_kb")
    g.add_edge("retrieve_kb", "suggest_patch")
    g.add_edge("suggest_patch", "diagnose")
    g.add_edge("diagnose", END)
    return g.compile()


def _none_if_missing(text: str) -> str | None:
    return None if text.startswith("<script for job") or text.startswith("ERROR") else text


def _template_diagnosis(state: DiagnosisState) -> str:
    job = state["job"]
    findings = state.get("findings") or []
    if not findings or findings[0]["kind"] == "unknown":
        return (
            f"**Cause** - no known failure signature in the logs for job {job['job_id']} "
            f"(state {job['state']}, exit {job['exit_code']}).\n\n"
            "**Fix** - read the end of stderr by hand, then see the general triage notes:\n"
            + _kb_text(state, limit=1)
        )
    top = findings[0]
    lines = [f"**Cause** - {top['summary']} (rule-based, confidence {top['confidence']:.0%})."]
    if len(findings) > 1 and findings[1]["confidence"] >= 0.6:
        lines.append(f"Also seen: {findings[1]['summary'].lower()}.")
    lines.append("\n**Evidence**")
    lines += [f"> {e}" for e in top["evidence"][:3]]
    lines.append("\n**Fix**")
    lines.append(_kb_text(state, limit=2))
    patch = state.get("patch")
    if patch and patch.get("note"):
        lines.append(f"\n**Script change** - {patch['note']}")
    return "\n".join(lines)


def _kb_text(state: DiagnosisState, limit: int) -> str:
    hits = state.get("kb_hits") or []
    if not hits:
        return "(knowledge base empty; run `slurmpilot kb build`)"
    useful = [h for h in hits if not h["section"].lower().startswith(("symptom", "overview"))] or hits
    return "\n\n".join(h["text"] for h in useful[:limit])


def diagnose_job(job_id: str, client: ToolClient | None = None, kb: KnowledgeBase | None = None, use_llm: bool = True, direct: bool = False) -> dict:
    own_client = client is None
    client = client or get_tool_client(direct=direct)
    try:
        graph = build_graph(client, kb, use_llm=use_llm)
        result = graph.invoke({"job_id": str(job_id)})
    finally:
        if own_client:
            client.close()
    return dict(result)


def diagnosis_to_json(result: dict) -> str:
    return json.dumps({k: v for k, v in result.items() if k not in ("stderr", "stdout")}, indent=2)