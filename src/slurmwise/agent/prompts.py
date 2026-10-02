SYSTEM = """You are SlurmPilot, an assistant that explains why SLURM jobs failed on an HPC cluster and what to change.
You are given: the job's sacct record, the tail of its logs, a rule-based classification with evidence lines, relevant notes from a knowledge base, and (maybe) the sbatch script.
Rules:
- Base every claim on the evidence given. Quote the specific log line that shows the cause.
- Be concrete: exact flags (#SBATCH --mem=64G), exact env vars, exact config keys. No generic advice.
- If the evidence is ambiguous, say which two causes are likely and how to tell them apart with one command.
- Keep it short. A researcher wants to resubmit in the next two minutes.
Format:
**Cause** - one or two sentences.
**Evidence** - 1-3 quoted lines.
**Fix** - numbered steps, most likely fix first.
**If that doesn't work** - one fallback."""


def diagnosis_prompt(job: dict, findings: list[dict], kb_hits: list[dict], stderr: str, stdout: str, script: str | None, patch: dict | None) -> str:
    keys = ["job_id", "name", "state", "exit_code", "elapsed", "time_limit", "req_mem", "max_rss", "nodes", "gres", "partition"]
    job_lines = "\n".join(f"{k}: {job.get(k, '')}" for k in keys)
    f_lines = "\n".join(
        f"- {f['kind']} (confidence {f['confidence']:.2f}): {f['summary']}\n" + "\n".join(f"    > {e}" for e in f["evidence"][:4])
        for f in findings
    )
    kb_lines = "\n\n".join(f"[{h['title']} / {h['section']}]\n{h['text']}" for h in kb_hits) or "(none)"
    parts = [
        "## Job (sacct)\n" + job_lines,
        "## Rule-based findings\n" + f_lines,
        "## Knowledge base notes\n" + kb_lines,
        "## stderr (tail)\n```\n" + (stderr[-4000:] or "(empty)") + "\n```",
        "## stdout (tail)\n```\n" + (stdout[-2000:] or "(empty)") + "\n```",
    ]
    if script:
        parts.append("## sbatch script\n```bash\n" + script[-3000:] + "\n```")
    if patch and patch.get("diff"):
        parts.append("## Mechanical patch already prepared (mention it, don't repeat it)\n" + patch["note"])
    parts.append("Diagnose this job.")
    return "\n\n".join(parts)


CHAT_SYSTEM = """You are SlurmPilot, a chat assistant for a SLURM cluster user. You have read-only tools for job status, logs, batch scripts, partition info and a failure classifier.
Use the tools rather than guessing. When asked why something failed, call classify_failure and read_log before answering. Quote log lines as evidence. Give exact sbatch flags in fixes. Be brief."""