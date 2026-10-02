from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from .llm import get_llm
from .prompts import CHAT_SYSTEM
from .tools import ToolClient, as_langchain_tools


def build_chat_agent(client: ToolClient):
    llm = get_llm(True)
    if llm is None:
        raise RuntimeError("Where is the LLM endpoint?")
    tools = as_langchain_tools(client)
    return create_react_agent(llm, tools, prompt=CHAT_SYSTEM)


def chat_once(agent, history: list, text: str) -> tuple[str, list]:
    history = history + [("user", text)]
    out = agent.invoke({"messages": history})
    msgs = out["messages"]
    reply = msgs[-1].content if msgs else ""
    return reply, msgs