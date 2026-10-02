from slurmwise.eval import (eval_classifier, eval_retrieval, load_benchmark)


def test_benchmark_loads():
    ex = load_benchmark()
    assert len(ex) >= 20
    assert all(e.kind and e.doc and e.log for e in ex)


def test_classifier_accuracy_is_high():
    ex = load_benchmark()
    r = eval_classifier(ex)
    assert r.accuracy >= 0.85
    assert r.n == len(ex)


def test_retrieval_finds_right_doc():
    ex = load_benchmark()
    r = eval_retrieval(ex, embedding="hash")
    assert r.recall_at_1 >= 0.8
    assert r.recall_at_3 >= r.recall_at_1
    assert 0 <= r.mrr <= 1


def test_retrieval_report_summary():
    r = eval_retrieval(load_benchmark(), embedding="hash")
    s = r.summary()
    assert set(s) == {"embedding", "n", "recall@1", "recall@3", "mrr"}
