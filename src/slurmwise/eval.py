from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import JobInfo
from .classify import classify
from .kb import KnowledgeBase

BENCHMARK = Path(__file__).resolve().parent / "benchmark" / "labeled_failures.json"


@dataclass
class Example:
    id: str
    kind: str
    doc: str
    state: str
    log: str


def load_benchmark(path: Path = BENCHMARK) -> list[Example]:
    with open(path) as f:
        return [Example(**d) for d in json.load(f)]


# ---------- classifier ----------
@dataclass
class ClassifierReport:
    n: int
    correct: int
    confusions: list[tuple[str, str, str]] = field(default_factory=list)  # (id, expected, got)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def eval_classifier(examples: list[Example]) -> ClassifierReport:
    correct = 0
    confusions = []
    for ex in examples:
        job = JobInfo(ex.id, state=ex.state)
        findings = classify(job, stderr=ex.log, stdout="")
        got = findings[0].kind if findings else "none"
        if got == ex.kind:
            correct += 1
        else:
            confusions.append((ex.id, ex.kind, got))
    return ClassifierReport(n=len(examples), correct=correct, confusions=confusions)


# ---------- retrieval ----------
def _rank_of_doc(hits, doc_id: str) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h.doc_id.split("::")[0] == doc_id:
            return i
    return None


@dataclass
class RetrievalReport:
    n: int
    embedding: str
    recall_at_1: float
    recall_at_3: float
    mrr: float
    misses: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "embedding": self.embedding,
            "n": self.n,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "mrr": round(self.mrr, 4),
        }


def eval_retrieval(examples: list[Example], embedding: str = "hash", k: int = 5) -> RetrievalReport:
    kb = KnowledgeBase(in_memory=True, embedding=embedding)
    kb.build()
    r1 = r3 = 0
    rr_sum = 0.0
    misses = []
    for ex in examples:
        job = JobInfo(ex.id, state=ex.state)
        findings = classify(job, stderr=ex.log, stdout="")
        kinds = [f.kind for f in findings[:2]]
        query = ex.log if not findings else (findings[0].summary + " " + ex.log)
        hits = kb.query(query, kinds=kinds, k=k)
        rank = _rank_of_doc(hits, ex.doc)
        if rank == 1:
            r1 += 1
        if rank is not None and rank <= 3:
            r3 += 1
        rr_sum += (1.0 / rank) if rank else 0.0
        if rank is None:
            misses.append(ex.id)
    n = len(examples)
    return RetrievalReport(n, embedding, r1 / n, r3 / n, rr_sum / n, misses)


def ablation(examples: list[Example], embeddings: list[str] = ("hash", "st"), k: int = 5) -> list[RetrievalReport]:
    out = []
    for emb in embeddings:
        try:
            out.append(eval_retrieval(examples, embedding=emb, k=k))
        except Exception as e:  # sentence-transformers may be unavailable offline
            out.append(RetrievalReport(len(examples), f"{emb} (unavailable: {type(e).__name__})", 0, 0, 0))
    return out