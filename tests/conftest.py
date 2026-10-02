import os

import pytest

os.environ["SLURMPILOT_BACKEND"] = "mock"
os.environ["SLURMPILOT_EMBEDDINGS"] = "hash"
os.environ["SLURMPILOT_NO_LLM"] = "1"


@pytest.fixture(scope="session")
def backend():
    from slurmwise.backends import get_backend

    return get_backend("mock")


@pytest.fixture(scope="session")
def kb():
    from slurmwise.kb import KnowledgeBase

    k = KnowledgeBase(in_memory=True, embedding="hash")
    k.build()
    return k


@pytest.fixture(scope="session")
def ds_jobs(backend):
    return backend.history()
