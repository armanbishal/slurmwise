from slurmwise.kb import load_documents


def test_docs_have_front_matter():
    chunks = load_documents()
    assert len(chunks) > 20
    kinds = {k for c in chunks for k in c["metadata"]["kinds"].split(",")}
    for k in ["oom", "cuda_oom", "time_limit", "cuda_error", "nccl", "import_error", "node_fail", "disk_quota", "unknown"]:
        assert k in kinds, k


def test_query_prefers_matching_kind(kb):
    hits = kb.query("job cancelled DUE TO TIME LIMIT", kinds=["time_limit"], k=2)
    assert hits[0].title.startswith("Wall-clock")
    hits = kb.query("ModuleNotFoundError No module named unsloth", kinds=["import_error"], k=2)
    assert "Missing Python module" in hits[0].title
