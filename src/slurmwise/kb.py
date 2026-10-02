from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

KNOWLEDGE_DIR = Path(os.environ.get("SLURMPILOT_KNOWLEDGE", Path(__file__).resolve().parent / "knowledge"))
DEFAULT_PERSIST = Path(os.environ.get("SLURMPILOT_HOME", Path.home() / ".slurmpilot")) / "chroma"


# one retrieved knowledge-base chunk and how well it matched
@dataclass
class KBHit:
    doc_id: str
    title: str
    section: str
    text: str
    kinds: list[str]
    score: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# read the knowledge markdown and split each file into section chunks
def load_documents(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[dict]:
    chunks = []
    for path in sorted(knowledge_dir.glob("*.md")):
        text = path.read_text()
        meta, body = _split_front_matter(text)
        doc_id = meta.get("id", path.stem)
        title = meta.get("title", path.stem)
        kinds = [str(k) for k in meta.get("kinds", [])]
        tags = [str(t) for t in meta.get("tags", [])]
        parts = re.split(r"(?m)^## ", body)
        head = parts[0].strip()
        sections = [("overview", head)] if head else []
        for p in parts[1:]:
            name, _, rest = p.partition("\n")
            sections.append((name.strip(), rest.strip()))
        for i, (name, sec_text) in enumerate(sections):
            if not sec_text:
                continue
            chunks.append(
                {
                    "id": f"{doc_id}::{i}",
                    "text": f"{title} - {name}\n{sec_text}",
                    "metadata": {
                        "doc_id": doc_id,
                        "title": title,
                        "section": name,
                        "kinds": ",".join(kinds),
                        "tags": ",".join(tags),
                    },
                }
            )
    return chunks


def _split_front_matter(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    return yaml.safe_load(m.group(1)) or {}, m.group(2)


class HashEmbedding:

    dim = 512

    def is_legacy(self) -> bool:
        return False

    @staticmethod
    def name() -> str:
        return "slurmpilot_hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()

    def __call__(self, input):  # chroma passes a list of strings
        out = []
        for text in input:
            vec = [0.0] * self.dim
            for tok in re.findall(r"[a-z0-9_]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_query(self, input):
        return self(input)

    def embed_documents(self, input):
        return self(input)


def _embedding_function(kind: str):
    kind = kind or os.environ.get("SLURMPILOT_EMBEDDINGS", "auto")
    if kind in ("auto", "st", "sentence-transformers"):
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            return SentenceTransformerEmbeddingFunction(
                model_name=os.environ.get("SLURMPILOT_EMBED_MODEL", "all-MiniLM-L6-v2")
            )
        except Exception:
            if kind != "auto":
                raise
    return HashEmbedding()


class KnowledgeBase:
    def __init__(self, persist_dir: str | os.PathLike | None = None, embedding: str = "auto", in_memory: bool = False):
        import chromadb

        if in_memory:
            self.client = chromadb.EphemeralClient()
        else:
            persist_dir = Path(persist_dir or DEFAULT_PERSIST)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.ef = _embedding_function(embedding)
        self.collection = self._open_collection()

    def _open_collection(self):
        kwargs = dict(name="slurm_failures", embedding_function=self.ef, metadata={"hnsw:space": "cosine"})
        try:
            return self.client.get_or_create_collection(**kwargs)
        except ValueError:
            try:
                self.client.delete_collection("slurm_failures")
            except Exception:
                pass
            return self.client.create_collection(**kwargs)

    def build(self, knowledge_dir: Path = KNOWLEDGE_DIR) -> int:
        chunks = load_documents(knowledge_dir)
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def ensure_built(self) -> None:
        if self.collection.count() == 0:
            self.build()

    def query(self, text: str, kinds: list[str] | None = None, k: int = 4) -> list[KBHit]:
        self.ensure_built()
        n = min(k * 3, max(self.collection.count(), 1))
        res = self.collection.query(query_texts=[text], n_results=n)
        hits = []
        for doc_id, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            doc_kinds = [x for x in meta.get("kinds", "").split(",") if x]
            score = 1.0 - float(dist)
            if kinds and any(kd in kinds for kd in doc_kinds):
                score += 0.5
            hits.append(KBHit(doc_id, meta["title"], meta["section"], doc, doc_kinds, round(score, 4)))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]
