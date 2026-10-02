import os
import shutil

from .base import JobInfo, SlurmBackend
from .mock import MockBackend
from .real import RealBackend


def get_backend(kind: str | None = None) -> SlurmBackend:
    """Pick a backend.

    Order: explicit arg, SLURMPILOT_BACKEND env var, then auto-detect (real if `sacct`
    is on PATH, mock otherwise). The mock reads fixtures from SLURMPILOT_FIXTURES or the
    bundled test fixtures.
    """
    kind = kind or os.environ.get("SLURMPILOT_BACKEND") or ("real" if shutil.which("sacct") else "mock")
    if kind == "real":
        return RealBackend()
    if kind == "mock":
        return MockBackend(os.environ.get("SLURMPILOT_FIXTURES"))
    raise ValueError(f"unknown backend {kind!r}")


__all__ = ["JobInfo", "SlurmBackend", "MockBackend", "RealBackend", "get_backend"]
