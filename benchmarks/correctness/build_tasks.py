#!/usr/bin/env python3
"""
Build the FROZEN correctness test set from REAL repo data.

Three families of context, all mined from this repository (no synthetic
payloads):
  - prompt  : real markdown docs + real Rust source files
  - history : real git commits turned into a multi-turn conversation; the
              question targets an EARLY turn (long-range recall — exactly
              what compress_history must preserve)
  - output  : real stdout/stderr of real shell commands run now

For each context, a strong judge model (sonnet) extracts a question + a set of
atomic key facts, derived from the COMPLETE context before any compression.
Facts are frozen into tasks.json so the run never regenerates them and all
three arms are scored against the identical fact set.

Usage:
    python3 build_tasks.py            # full build (~60 tasks, ~60 sonnet calls)
    python3 build_tasks.py --limit 6  # smoke: a couple of each family
"""

from __future__ import annotations

import argparse
import json
import subprocess

import lib

REPO = lib.REPO
OUT = lib.HERE / "tasks.json"

GENFACTS_SYSTEM = """You build a fact-recall test item from a CONTEXT block.

Return STRICT JSON only, no prose, no markdown fence:
{"question": "...", "key_facts": ["...", "..."]}

Rules:
- The QUESTION must be answerable using ONLY the CONTEXT, and must REQUIRE
  specific information from it (not answerable by generic knowledge).
- KEY_FACTS: 4 to 8 atomic facts. Each fact must be:
    * explicitly present in the CONTEXT (never inferred or invented),
    * necessary for a complete, correct answer to the QUESTION,
    * independently checkable (one claim per fact),
    * concrete (names, numbers, identifiers, error codes, file paths — not vague).
- Do NOT include any fact that is not literally supported by the CONTEXT.
%s
CONTEXT:
---
%s
---"""

HINTS = {
    "prompt": "- The question should target a SPECIFIC section/detail, not the whole document.",
    "history": "- The question MUST require recalling details from the EARLY messages of the conversation (long-range recall).",
    "output": "- The question should target a buried specific detail: an error code, a count, a warning, or a file path.",
}


def gen_item(context: str, op: str):
    prompt = GENFACTS_SYSTEM % (HINTS[op], context[:24000])
    raw = lib.claude(prompt, model="sonnet", kind="genfacts", timeout=240)
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip() if "```" in raw else raw
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # last-ditch: grab the outermost JSON object
        try:
            d = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            return None
    q = (d.get("question") or "").strip()
    facts = [
        f.strip()
        for f in (d.get("key_facts") or [])
        if isinstance(f, str) and f.strip()
    ]
    if not q or len(facts) < 3:
        return None
    return {"question": q, "key_facts": facts}


# ---------------- Context collectors ----------------
def windows(text: str, target_tok=2200, max_windows=99):
    """Split text into ~target_tok windows on paragraph boundaries."""
    paras = text.split("\n\n")
    out, cur = [], ""
    for p in paras:
        cur = (cur + "\n\n" + p) if cur else p
        if lib.count_tokens(cur) >= target_tok:
            out.append(cur)
            cur = ""
        if len(out) >= max_windows:
            break
    if cur.strip() and len(out) < max_windows:
        out.append(cur)
    return out


def collect_prompt_contexts():
    ctxs = []
    docs = [
        "README.md",
        "BENCHMARK.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
    ]
    doc_budget = {
        "README.md": 3,
        "BENCHMARK.md": 2,
        "CHANGELOG.md": 5,
        "CONTRIBUTING.md": 1,
        "CODE_OF_CONDUCT.md": 1,
    }
    for d in docs:
        p = REPO / d
        if not p.exists():
            continue
        for i, w in enumerate(windows(p.read_text(), 2200, doc_budget.get(d, 2))):
            if lib.count_tokens(w) >= 400:
                ctxs.append({"label": f"doc:{d}#{i}", "text": w})
    # real source files: a spread of crates
    src = sorted((REPO / "sophon" / "crates").rglob("*.rs"))
    src = [f for f in src if "test" not in f.name and "/tests/" not in str(f)]
    picked, seen_crate = [], {}
    for f in src:
        crate = f.relative_to(REPO / "sophon" / "crates").parts[0]
        n = lib.count_tokens(f.read_text())
        if 600 <= n <= 3500 and seen_crate.get(crate, 0) < 3:
            picked.append(f)
            seen_crate[crate] = seen_crate.get(crate, 0) + 1
    for f in picked[:14]:
        ctxs.append({"label": f"src:{f.relative_to(REPO)}", "text": f.read_text()})
    return ctxs


def git_commits(n=64):
    hashes = subprocess.run(
        ["git", "-C", str(REPO), "log", "--no-merges", f"-{n}", "--pretty=%H"],
        capture_output=True,
        text=True,
    ).stdout.split()
    rows = []
    for h in hashes:
        # one show per commit: %x1f-delimited subject/body, then the --stat block
        raw = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "show",
                "--stat",
                "--pretty=%h%x1f%s%x1f%b%x1e",
                h,
            ],
            capture_output=True,
            text=True,
        ).stdout
        meta, _, stat = raw.partition("\x1e")
        parts = meta.split("\x1f")
        if len(parts) < 2:
            continue
        rows.append(
            {
                "hash": parts[0].strip(),
                "subject": parts[1].strip(),
                "body": (parts[2] if len(parts) > 2 else "").strip(),
                "stat": stat.strip(),
            }
        )
    return rows


def collect_history_contexts(n_hist=20):
    commits = git_commits(64)
    if len(commits) < 12:
        return []
    ctxs = []
    win = 9  # turns per conversation
    step = max(1, (len(commits) - win) // n_hist)
    for k in range(n_hist):
        start = k * step
        block = commits[start : start + win]
        if len(block) < 6:
            break
        messages = []
        for c in block:
            messages.append(
                {"role": "user", "content": f"Walk me through commit {c['hash']}."}
            )
            asst = f"{c['subject']}\n\n{c['body']}\n\nFiles:\n{c['stat']}".strip()
            messages.append({"role": "assistant", "content": asst})
        # context text used for fact extraction = the full conversation
        text = "\n".join(f"{m['role'].title()}: {m['content']}" for m in messages)
        ctxs.append(
            {
                "label": f"hist:{block[0]['hash']}..{block[-1]['hash']}",
                "text": text,
                "messages": messages,
            }
        )
    return ctxs


SHELL_CMDS = [
    ("cargo build --release -p prompt-compressor", "build a crate"),
    ("cargo clippy -p prompt-compressor 2>&1", "clippy warnings"),
    ("cargo tree -p mcp-integration --depth 1", "dependency tree"),
    ("git log --oneline -25", "recent history"),
    ("git log --stat -3", "recent diffs"),
    ("ls -laR sophon/crates/prompt-compressor", "directory listing"),
    ("wc -l sophon/crates/*/src/*.rs", "line counts"),
    ("cargo metadata --no-deps --format-version 1", "workspace metadata"),
    ("git diff HEAD~5 HEAD --stat", "5-commit diffstat"),
    ("git show --stat HEAD", "head commit"),
    ("cargo test -p prompt-compressor 2>&1", "test run"),
    ("du -sh sophon/target/release/sophon sophon/target/release", "artifact sizes"),
    ("grep -rn 'pub fn' sophon/crates/prompt-compressor/src", "public fns"),
    ("git remote -v && git branch -a", "remotes and branches"),
    ("head -50 Cargo.toml sophon/Cargo.toml", "manifest heads"),
]


def collect_output_contexts(limit=15):
    ctxs = []
    for cmd, desc in SHELL_CMDS[:limit]:
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            continue
        if lib.count_tokens(output) < 120:
            continue
        ctxs.append({"label": f"cmd:{desc}", "command": cmd, "text": output[:18000]})
    return ctxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit", type=int, default=0, help="max contexts per family (smoke)"
    )
    args = ap.parse_args()

    families = {
        "prompt": collect_prompt_contexts(),
        "history": collect_history_contexts(),
        "output": collect_output_contexts(),
    }
    tasks = []
    tid = 0
    for op, ctxs in families.items():
        if args.limit:
            ctxs = ctxs[: args.limit]
        print(f"[{op}] {len(ctxs)} contexts")
        for c in ctxs:
            item = gen_item(c["text"], op)
            if not item:
                print(f"   skip {c['label']} (no item)")
                continue
            tid += 1
            task = {
                "id": f"{op}-{tid:03d}",
                "op": op,
                "label": c["label"],
                "question": item["question"],
                "key_facts": item["key_facts"],
            }
            if op == "prompt":
                task["prompt"] = c["text"]
                task["query"] = item["question"]
            elif op == "history":
                task["messages"] = c["messages"]
            elif op == "output":
                task["command"] = c["command"]
                task["output"] = c["text"]
            tasks.append(task)
            print(
                f"   + {task['id']:<12} {c['label']:<40} facts={len(item['key_facts'])}"
            )

    OUT.write_text(json.dumps({"tasks": tasks}, indent=2))
    print(f"\n{len(tasks)} tasks -> {OUT}")
    by_op = {}
    for t in tasks:
        by_op[t["op"]] = by_op.get(t["op"], 0) + 1
    print("by op:", by_op)


if __name__ == "__main__":
    main()
