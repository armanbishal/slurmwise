from __future__ import annotations

import os

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"


def llm_settings() -> dict:
    return {
        "base_url": os.environ.get("SLURMPILOT_LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": os.environ.get("SLURMPILOT_LLM_MODEL", DEFAULT_MODEL),
        "api_key": os.environ.get("SLURMPILOT_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "ollama")),
        "temperature": float(os.environ.get("SLURMPILOT_LLM_TEMPERATURE", "0.1")),
    }


def get_llm(enabled: bool = True):
    if not enabled or os.environ.get("SLURMPILOT_NO_LLM") == "1":
        return None
    from langchain_openai import ChatOpenAI

    s = llm_settings()
    return ChatOpenAI(
        base_url=s["base_url"],
        api_key=s["api_key"],
        model=s["model"],
        temperature=s["temperature"],
        timeout=120,
        max_retries=1,
    )