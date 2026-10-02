from __future__ import annotations

import json
import logging
import os
import time

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent.graph import build_graph, diagnosis_to_json
from .agent.tools import get_tool_client, loads_or_text
from .kb import KnowledgeBase

app = typer.Typer(help="Diagnose failed SLURM jobs.", no_args_is_help=True, add_completion=False)
kb_app = typer.Typer(help="Knowledge base commands.")
app.add_typer(kb_app, name="kb")
console = Console()

for _name in ("httpx", "huggingface_hub", "sentence_transformers", "chromadb", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _kb(embedding: str | None = None) -> KnowledgeBase:
    kb = KnowledgeBase(embedding=embedding or os.environ.get("SLURMPILOT_EMBEDDINGS", "auto"))
    kb.ensure_built()
    return kb


def _print_diagnosis(result: dict, show_patch: bool = True) -> None:
    job = result.get("job")
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)
    title = f"job {job['job_id']}  {job['name']}  [{job['state']}]  exit {job['exit_code']}  {job['elapsed']}/{job['time_limit']}"
    console.print(Panel(Markdown(result.get("diagnosis", "")), title=title, subtitle="llm" if result.get("used_llm") else "rules + kb"))
    patch = result.get("patch")
    if show_patch and patch and patch.get("diff"):
        console.print(Panel(patch["diff"], title="suggested sbatch change", subtitle=patch.get("note", "")))


@app.command()
def diagnose(
    job_id: str = typer.Argument(..., help="SLURM job id"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the model, print the rule-based diagnosis only"),
    direct: bool = typer.Option(False, "--direct", help="Call tools in-process instead of over MCP stdio"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    embedding: str = typer.Option(None, help="Chroma embedding: auto | st | hash (default: $SLURMPILOT_EMBEDDINGS or auto)"),
):
    client = get_tool_client(direct=direct)
    try:
        graph = build_graph(client, _kb(embedding), use_llm=not no_llm)
        result = dict(graph.invoke({"job_id": job_id}))
    finally:
        client.close()
    if as_json:
        print(diagnosis_to_json(result))
        return
    _print_diagnosis(result)


@app.command()
def jobs(
    user: str = typer.Option(None, help="Defaults to $USER"),
    state: str = typer.Option(None, help="Comma-separated states, e.g. FAILED,TIMEOUT,OUT_OF_MEMORY"),
    limit: int = typer.Option(20),
    direct: bool = typer.Option(False, "--direct"),
):
    client = get_tool_client(direct=direct)
    try:
        rows = loads_or_text(client.call("list_jobs", user=user, states=state, limit=limit))
    finally:
        client.close()
    t = Table("job", "name", "state", "exit", "elapsed", "limit", "mem", "nodes")
    for j in rows:
        color = {"FAILED": "red", "TIMEOUT": "yellow", "OUT_OF_MEMORY": "red", "NODE_FAIL": "magenta", "RUNNING": "green", "COMPLETED": "dim"}.get(
            j["state"].split()[0], "white"
        )
        t.add_row(j["job_id"], j["name"], f"[{color}]{j['state']}[/{color}]", j["exit_code"], j["elapsed"], j["time_limit"], j["req_mem"], j["nodes"])
    console.print(t)


@app.command()
def watch(
    user: str = typer.Option(None, help="Defaults to $USER"),
    interval: int = typer.Option(60, help="Seconds between polls"),
    no_llm: bool = typer.Option(False, "--no-llm"),
    direct: bool = typer.Option(False, "--direct"),
    once: bool = typer.Option(False, "--once", help="Poll one time and exit (for cron / tests)"),
):
    client = get_tool_client(direct=direct)
    graph = build_graph(client, _kb(), use_llm=not no_llm)
    seen: set[str] = set()
    bad = "FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL"
    try:
        first = True
        while True:
            rows = loads_or_text(client.call("list_jobs", user=user, states=bad, limit=50))
            new = [j for j in rows if j["job_id"] not in seen]
            if first and not once:
                # don't spam with history on startup, only report what fails from now on
                seen.update(j["job_id"] for j in rows)
                console.print(f"watching {len(rows)} already-failed jobs ignored; waiting for new failures (every {interval}s)")
                first = False
            else:
                for j in new:
                    seen.add(j["job_id"])
                    _print_diagnosis(dict(graph.invoke({"job_id": j["job_id"]})))
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


@app.command()
def chat(direct: bool = typer.Option(False, "--direct")):
    from .agent.chat import build_chat_agent, chat_once

    client = get_tool_client(direct=direct)
    try:
        agent = build_chat_agent(client)
        history: list = []
        console.print("[dim]slurmpilot chat. ctrl-c to quit.[/dim]")
        while True:
            try:
                text = console.input("[bold]you>[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            reply, history = chat_once(agent, history, text)
            console.print(Markdown(reply))
    finally:
        client.close()


@app.command("serve-mcp")
def serve_mcp():
    from .mcp_server import main

    main()


@app.command()
def stats(
    user: str = typer.Option(None, help="Defaults to $USER"),
    since: str = typer.Option("now-30days", help="sacct time window, e.g. now-30days, now-7days, 2026-01-01"),
    gpu_watts: float = typer.Option(400.0, help="assumed power per GPU, for the energy estimate"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Fleet analytics over your job history: failure rate, wasted GPU-hours, and the
    energy that waste cost. Read-only. Failed jobs still burn the power they were
    allocated, so this quantifies that."""
    from .analytics import EnergyModel, compute_stats
    from .backends import get_backend

    jobs = get_backend().history(user=user, since=since)
    st = compute_stats(jobs, EnergyModel(gpu_watts=gpu_watts))

    if as_json:
        print(json.dumps(st.summary(), indent=2))
        return

    if st.total_jobs == 0:
        console.print(f"[yellow]No jobs found for the window {since}.[/yellow]")
        return

    head = Table.grid(padding=(0, 2))
    head.add_row("Jobs", str(st.total_jobs))
    head.add_row("Failed", f"[red]{st.failed_jobs}[/red]  ({st.failure_rate:.0%} failure rate)")
    head.add_row("Completed", str(st.completed_jobs))
    head.add_row("Cancelled / preempted", str(st.soft_jobs))
    head.add_row("Wasted GPU-hours", f"{st.wasted_gpu_hours:.1f}")
    head.add_row("Wasted CPU-hours", f"{st.wasted_cpu_hours:.1f}")
    head.add_row("Wasted energy", f"[red]{st.wasted_kwh:.1f} kWh[/red]  (~{st.wasted_co2_kg:.1f} kg CO2)")
    console.print(Panel(head, title=f"SLURM usage, {since} to now", subtitle="failed jobs = wasted compute = wasted energy"))

    if st.by_failure_kind:
        t = Table("failure", "count", title="Failures by kind")
        for kind, n in st.by_failure_kind.most_common():
            t.add_row(kind, str(n))
        console.print(t)

    risky = [(p, f / tot, f, tot) for p, (f, tot) in st.by_partition.items() if tot >= 1]
    risky.sort(key=lambda x: -x[1])
    if risky:
        t = Table("partition", "failure rate", "failed / total", "wasted kWh", title="By partition")
        for p, rate, f, tot in risky:
            t.add_row(p, f"{rate:.0%}", f"{f} / {tot}", f"{st.wasted_kwh_by_partition.get(p, 0):.1f}")
        console.print(t)


@app.command()
def eval(
    embedding: str = typer.Option("hash", help="retriever embedding to score: hash (lexical) or st (semantic)"),
    semantic_ablation: bool = typer.Option(False, "--semantic-ablation", help="also run the semantic embedder and compare (slower, needs sentence-transformers)"),
    as_json: bool = typer.Option(False, "--json"),
):
    from .eval import ablation, eval_classifier, eval_retrieval, load_benchmark

    examples = load_benchmark()
    clf = eval_classifier(examples)
    ret = eval_retrieval(examples, embedding=embedding)

    if as_json:
        out = {"classifier_accuracy": round(clf.accuracy, 4), "n": clf.n, "retrieval": ret.summary()}
        if semantic_ablation:
            out["ablation"] = [r.summary() for r in ablation(examples)]
        print(json.dumps(out, indent=2))
        return

    g = Table.grid(padding=(0, 2))
    g.add_row("Benchmark size", str(clf.n))
    g.add_row("Classifier accuracy", f"{clf.accuracy:.0%}  ({clf.correct}/{clf.n})")
    g.add_row("Retrieval Recall@1", f"{ret.recall_at_1:.0%}")
    g.add_row("Retrieval Recall@3", f"{ret.recall_at_3:.0%}")
    g.add_row("Retrieval MRR", f"{ret.mrr:.3f}")
    console.print(Panel(g, title="Diagnosis pipeline evaluation", subtitle=f"retriever: {ret.embedding}"))

    if clf.confusions:
        t = Table("case", "expected", "got", title="Classifier misses")
        for cid, exp, got in clf.confusions:
            t.add_row(cid, exp, got)
        console.print(t)

    if semantic_ablation:
        t = Table("embedding", "recall@1", "recall@3", "mrr", title="Retrieval ablation: lexical vs semantic")
        for r in ablation(examples):
            t.add_row(r.embedding, f"{r.recall_at_1:.2f}", f"{r.recall_at_3:.2f}", f"{r.mrr:.3f}")
        console.print(t)


@app.command()
def tools(direct: bool = typer.Option(False, "--direct")):
    client = get_tool_client(direct=direct)
    try:
        for t in client.list_tools():
            console.print(f"[bold]{t['name']}[/bold]  {t['description']}")
            console.print("   args: " + ", ".join(t["input_schema"].get("properties", {}).keys()))
    finally:
        client.close()


@kb_app.command("build")
def kb_build(embedding: str = typer.Option(None, help="auto | st | hash")):
    kb = KnowledgeBase(embedding=embedding or os.environ.get("SLURMPILOT_EMBEDDINGS", "auto"))
    n = kb.build()
    console.print(f"indexed {n} chunks into {kb.client.get_settings().persist_directory or 'memory'}")


@kb_app.command("query")
def kb_query(text: str, k: int = 3, embedding: str = None):
    for h in _kb(embedding).query(text, k=k):
        console.print(Panel(h.text, title=f"{h.title} / {h.section}", subtitle=f"score {h.score}"))


@app.command()
def version():
    print(__version__)


if __name__ == "__main__":
    app()