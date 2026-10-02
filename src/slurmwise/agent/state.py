from typing import TypedDict


class DiagnosisState(TypedDict, total=False):
    job_id: str
    job: dict | None
    stderr: str
    stdout: str
    script: str | None
    findings: list[dict]
    kb_hits: list[dict]
    patch: dict | None
    diagnosis: str
    used_llm: bool
    error: str | None
