#!/usr/bin/env python3
"""
End-to-end correctness bench.

For each frozen task, three arms at COMPARABLE token budget:
  oracle   — full context (quality ceiling)
  sophon   — Sophon-compressed context at budget B
  truncate — raw context head-truncated to the SAME token count Sophon produced

A weak answerer (haiku) answers each; a stronger, BLIND judge (sonnet) scores
key-fact recall (it sees only question + key_facts + answer, never the context
or which arm produced it). The headline is recall(sophon) - recall(truncate)
at equal budget, with a paired bootstrap CI.

Usage:
    python3 run_correctness.py                 # full run (cached/resumable)
    python3 run_correctness.py --limit 4       # smoke
    python3 run_correctness.py --workers 8 --ratio 3
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import lib

TASKS = lib.HERE / "tasks.json"
RES = lib.HERE / "results"
MIN_BUDGET = 256
ANSWER_MODEL = "haiku"
JUDGE_MODEL = "sonnet"

ANSWER_TMPL = """Answer the QUESTION using ONLY the CONTEXT below. Be concise and \
factual. If the context does not contain the information, say so plainly.

CONTEXT:
---
{ctx}
---

QUESTION: {q}

Answer:"""

JUDGE_TMPL = """You grade an answer for factual recall. You are given a QUESTION, \
a numbered list of KEY FACTS a complete correct answer must convey, and a \
candidate ANSWER. For each key fact, decide whether the ANSWER correctly \
conveys it. Judge ONLY what the answer states — do not use outside knowledge, \
do not reward facts the answer omits.

Return STRICT JSON only: {{"verdicts": [true, false, ...]}} with exactly one \
boolean per key fact, in the same order.

QUESTION: {q}

KEY FACTS:
{facts}

ANSWER:
---
{ans}
---"""


# ---------------- per-task context build (local, fast) ----------------
def raw_context(task):
    if task["op"] == "prompt":
        return task["prompt"]
    if task["op"] == "history":
        return "\n".join(
            f"{m['role'].title()}: {m['content']}" for m in task["messages"]
        )
    return task["output"]


def build_arms(task, ratio):
    """Return {arm: (context_text, tokens)} for the three conditions."""
    raw = raw_context(task)
    raw_tok = lib.count_tokens(raw)
    budget = max(MIN_BUDGET, raw_tok // ratio)

    if task["op"] == "prompt":
        s_txt, s_tok = lib.compress_prompt(task["prompt"], task["query"], budget)
    elif task["op"] == "history":
        s_txt, s_tok = lib.compress_history(
            task["messages"], budget, recent_window=4, query=task["question"]
        )
    else:  # output — no budget param; its own filtering sets the budget
        s_txt, s_tok = lib.compress_output(task["command"], task["output"])

    # Naive baseline is op-appropriate: a long prompt / command output is
    # naively HEAD-capped (keep the start), whereas a conversation is naively
    # kept as a recent sliding window (TAIL) — the realistic thing
    # compress_history competes against by summarising the dropped tail.
    if task["op"] == "history":
        t_txt, t_tok = lib.truncate_to_tokens_tail(raw, s_tok)
    else:
        t_txt, t_tok = lib.truncate_to_tokens(raw, s_tok)
    return {
        "oracle": (raw, raw_tok),
        "sophon": (s_txt, s_tok),
        "truncate": (t_txt, t_tok),
    }, raw_tok


# ---------------- answer + judge ----------------
def answer(ctx, q):
    a = lib.claude(ANSWER_TMPL.format(ctx=ctx, q=q), model=ANSWER_MODEL, kind="answer")
    return a or ""


def judge(q, key_facts, ans):
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(key_facts))
    raw = lib.claude(
        JUDGE_TMPL.format(q=q, facts=facts, ans=ans), model=JUDGE_MODEL, kind="judge"
    )
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.index("{") :] if "{" in raw else raw
    try:
        d = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    v = d.get("verdicts")
    if not isinstance(v, list) or not v:
        return None
    # align length to fact count
    v = [bool(x) for x in v][: len(key_facts)]
    v += [False] * (len(key_facts) - len(v))
    return v


# ---------------- stats ----------------
def bootstrap_ci(deltas, iters=10000, seed=17):
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    return (means[int(0.025 * iters)], means[int(0.975 * iters)])


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--ratio", type=int, default=3, help="compress to raw_tokens // ratio"
    )
    ap.add_argument(
        "--summary-mode",
        choices=["default", "extractive", "legacy", "llm"],
        default="default",
        help="compress_history summariser. default/extractive = deterministic fact-preserving; "
        "legacy = old first-sentence heuristic (SOPHON_LEGACY_SUMMARY=1); "
        "llm = abstractive (SOPHON_LLM_CMD=claude -p --model haiku). Tags output files.",
    )
    args = ap.parse_args()

    # The summary mode is enforced via env vars that `sophon serve` (spawned by
    # lib.rpc, which inherits this process's environment) reads at compress time.
    import os

    for v in ("SOPHON_NO_LLM_SUMMARY", "SOPHON_LEGACY_SUMMARY", "SOPHON_LLM_CMD"):
        os.environ.pop(v, None)
    if args.summary_mode in ("extractive",):
        os.environ["SOPHON_NO_LLM_SUMMARY"] = "1"
    elif args.summary_mode == "legacy":
        os.environ["SOPHON_NO_LLM_SUMMARY"] = "1"
        os.environ["SOPHON_LEGACY_SUMMARY"] = "1"
    elif args.summary_mode == "llm":
        os.environ["SOPHON_LLM_CMD"] = "claude -p --model haiku"
    args.tag = "" if args.summary_mode == "default" else f".{args.summary_mode}"

    tasks = json.loads(TASKS.read_text())["tasks"]
    if args.limit:
        # keep a balanced slice across ops
        by_op = {}
        sel = []
        for t in tasks:
            by_op.setdefault(t["op"], 0)
            if by_op[t["op"]] < args.limit:
                sel.append(t)
                by_op[t["op"]] += 1
        tasks = sel
    print(
        f"[run] {len(tasks)} tasks x 3 arms  (answer={ANSWER_MODEL}, judge={JUDGE_MODEL})"
    )

    # Phase A: build all arm contexts (local Sophon calls)
    cells = []
    for t in tasks:
        arms, raw_tok = build_arms(t, args.ratio)
        for arm, (ctx, tok) in arms.items():
            cells.append(
                {
                    "task": t,
                    "arm": arm,
                    "ctx": ctx,
                    "tokens": tok,
                    "raw_tokens": raw_tok,
                }
            )
    print(f"[run] {len(cells)} cells built")

    # Phase B: answers in parallel (cached)
    def do_answer(c):
        c["answer"] = answer(c["ctx"], c["task"]["question"])
        return c

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        cells = list(ex.map(do_answer, cells))
    print("[run] answers done")

    # Phase C: blind judge in parallel (cached)
    def do_judge(c):
        c["verdicts"] = judge(
            c["task"]["question"], c["task"]["key_facts"], c["answer"]
        )
        return c

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        cells = list(ex.map(do_judge, cells))
    print("[run] judging done")

    # Assemble per-task records
    records = {}
    for c in cells:
        tid = c["task"]["id"]
        r = records.setdefault(
            tid,
            {
                "id": tid,
                "op": c["task"]["op"],
                "label": c["task"]["label"],
                "n_facts": len(c["task"]["key_facts"]),
                "raw_tokens": c["raw_tokens"],
                "arms": {},
            },
        )
        v = c["verdicts"]
        recall = (sum(v) / len(v)) if v else None
        r["arms"][c["arm"]] = {
            "tokens": c["tokens"],
            "recall": recall,
            "judged": v is not None,
        }

    write_report(records, args)


def write_report(records, args):
    recs = list(records.values())
    arms = ["oracle", "sophon", "truncate"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    # overall
    overall = {}
    for a in arms:
        rc = [r["arms"][a]["recall"] for r in recs if a in r["arms"]]
        tk = [r["arms"][a]["tokens"] for r in recs if a in r["arms"]]
        raw = [r["raw_tokens"] for r in recs if a in r["arms"]]
        sav = [1 - t / rw for t, rw in zip(tk, raw) if rw]
        overall[a] = {"recall": mean(rc), "savings": mean(sav)}

    # paired deltas (tasks where both arms judged)
    def paired(a, b):
        d = []
        for r in recs:
            ra, rb = (
                r["arms"].get(a, {}).get("recall"),
                r["arms"].get(b, {}).get("recall"),
            )
            if ra is not None and rb is not None:
                d.append(ra - rb)
        return d

    d_st = paired("sophon", "truncate")
    d_so = paired("sophon", "oracle")
    ci_st = bootstrap_ci(d_st)
    ci_so = bootstrap_ci(d_so)

    # per-op
    by_op = {}
    for r in recs:
        by_op.setdefault(r["op"], []).append(r)

    losses = [
        r
        for r in recs
        if r["arms"].get("sophon", {}).get("recall") is not None
        and r["arms"].get("truncate", {}).get("recall") is not None
        and r["arms"]["sophon"]["recall"] < r["arms"]["truncate"]["recall"]
    ]

    RES.mkdir(exist_ok=True)
    tag = getattr(args, "tag", "")
    (RES / f"correctness{tag}.json").write_text(
        json.dumps(
            {
                "config": {
                    "ratio": args.ratio,
                    "answer_model": ANSWER_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "n_tasks": len(recs),
                },
                "overall": overall,
                "delta_sophon_minus_truncate": {
                    "mean": mean(d_st),
                    "ci95": ci_st,
                    "n": len(d_st),
                },
                "delta_sophon_minus_oracle": {
                    "mean": mean(d_so),
                    "ci95": ci_so,
                    "n": len(d_so),
                },
                "records": recs,
            },
            indent=2,
        )
    )

    # markdown
    L = []
    L.append("# Sophon — correctness bench (key-fact recall)\n")
    L.append(
        f"Answerer **{ANSWER_MODEL}** · blind judge **{JUDGE_MODEL}** · "
        f"compress ratio 1/{args.ratio} · **N={len(recs)}** real-repo tasks.\n"
    )
    L.append(
        "Each arm answers from a context of comparable token budget; the judge scores "
        "what fraction of the pre-frozen key facts survive into the answer.\n"
    )
    L.append(
        "**Baselines.** `oracle` = full context (ceiling). `truncate` = naive cap at the "
        "*same token count Sophon emitted*: HEAD-truncation for `prompt`/`output` (keep the "
        "start), TAIL/recent-window for `history` (keep recent turns — the realistic sliding "
        "window compress_history claims to beat by summarising the dropped tail). History "
        "questions deliberately target OLD turns: that is compress_history's reason to exist, "
        "so it is the charitable test of its core value.\n"
    )
    L.append("## Headline\n")
    L.append("| Arm | Mean recall | Mean token savings |")
    L.append("|---|---|---|")
    for a in arms:
        L.append(
            f"| {a} | {overall[a]['recall'] * 100:.1f}% | {overall[a]['savings'] * 100:.1f}% |"
        )
    L.append("")
    L.append(
        f"**Sophon − truncate (equal budget): {mean(d_st) * 100:+.1f} pts** "
        f"(95% CI [{ci_st[0] * 100:+.1f}, {ci_st[1] * 100:+.1f}], n={len(d_st)})  "
    )
    L.append(
        f"**Sophon − oracle (loss vs full context): {mean(d_so) * 100:+.1f} pts** "
        f"(95% CI [{ci_so[0] * 100:+.1f}, {ci_so[1] * 100:+.1f}], n={len(d_so)})"
    )
    L.append("")
    L.append("## Per-op breakdown\n")
    L.append(
        "| op | n | recall oracle | recall sophon | recall truncate | sophon−trunc | sophon savings |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for op, rs in sorted(by_op.items()):

        def m(a):
            return mean([r["arms"].get(a, {}).get("recall") for r in rs])

        sav = mean(
            [
                1 - r["arms"]["sophon"]["tokens"] / r["raw_tokens"]
                for r in rs
                if "sophon" in r["arms"] and r["raw_tokens"]
            ]
        )
        L.append(
            f"| {op} | {len(rs)} | {m('oracle') * 100:.1f}% | {m('sophon') * 100:.1f}% | "
            f"{m('truncate') * 100:.1f}% | {(m('sophon') - m('truncate')) * 100:+.1f} | {sav * 100:.1f}% |"
        )
    L.append("")
    L.append(f"## Where Sophon loses to dumb truncation ({len(losses)}/{len(recs)})\n")
    if losses:
        L.append("| task | op | sophon | truncate |")
        L.append("|---|---|---|---|")
        for r in losses:
            L.append(
                f"| {r['id']} ({r['label']}) | {r['op']} | "
                f"{r['arms']['sophon']['recall'] * 100:.0f}% | {r['arms']['truncate']['recall'] * 100:.0f}% |"
            )
    else:
        L.append("_None — Sophon ≥ truncation on every task._")
    L.append("")
    L.append("## What this means\n")
    d_st_mean = mean(d_st) * 100
    d_so_mean = mean(d_so) * 100
    L.append(
        f"- **The {overall['sophon']['savings'] * 100:.0f}% token saving is real but not free**: "
        f"compressing to 1/{args.ratio} budget costs **{-d_so_mean:.0f} pts of key-fact recall** "
        f"vs the full context ({overall['oracle']['recall'] * 100:.0f}% → "
        f"{overall['sophon']['recall'] * 100:.0f}%). This is the cost the token-only headline hid."
    )
    # This line is data-driven: whether Sophon meaningfully beats naive
    # truncation at equal budget depends on whether the bootstrap CI for
    # the paired delta clears zero. Never hard-code the conclusion — the
    # whole point of the bench is to report what the numbers actually say.
    st_lo, st_hi = ci_st[0] * 100, ci_st[1] * 100

    def op_delta(op):
        rs = by_op.get(op, [])
        if not rs:
            return 0.0
        s = mean([r["arms"].get("sophon", {}).get("recall") for r in rs])
        t = mean([r["arms"].get("truncate", {}).get("recall") for r in rs])
        return (s - t) * 100

    if ci_st[0] > 0:
        # Characterize the edge from the data, not a frozen adjective: at
        # +5 it's "modest", at +20 it isn't. Per-op wins/losses are derived
        # from the actual deltas so the sentence can't drift out of date.
        size = "small" if d_st_mean < 8 else ("solid" if d_st_mean < 18 else "large")
        wins = [op for op in ("prompt", "history", "output") if op_delta(op) > 1.0]
        loses = [op for op in ("prompt", "history", "output") if op_delta(op) < -1.0]
        clause = f"it wins on {', '.join(f'`{o}`' for o in wins)}" if wins else ""
        if loses:
            clause += f" and still loses on {', '.join(f'`{o}`' for o in loses)}"
        L.append(
            f"- **At equal budget, Sophon beats naive truncation by {d_st_mean:+.1f} pts** "
            f"(95% CI [{st_lo:+.1f}, {st_hi:+.1f}] — excludes 0, so the edge is statistically "
            f"real on this task mix). The edge is {size} and uneven: {clause.strip()}."
        )
    else:
        L.append(
            f"- **At equal budget, Sophon ≈ naive truncation** ({d_st_mean:+.1f} pts, "
            f"95% CI [{st_lo:+.1f}, {st_hi:+.1f}] spans 0). The compression is not meaningfully "
            '"smarter" than a dumb cap on average.'
        )

    def op_recall(op):
        rs = by_op.get(op, [])
        return (
            mean([r["arms"].get("sophon", {}).get("recall") for r in rs]) * 100
            if rs
            else 0.0
        )

    L.append(
        "- **`compress_history` summariser** (`memory-manager/src/summarizer.rs`): the old default kept "
        "only the first sentence of each dropped message, so buried old facts vanished (recall ~0.8%). "
        "The current default is a deterministic **query-aware extractive** summariser: it ranks the "
        "dropped older lines by relevance to the question (then information density), so the line that "
        f"answers the query lands in the summary → history recall **{op_recall('history'):.1f}%**. "
        "See `COMPARISON.md`. (Pass the question via the `query` arg to activate it; with no query it "
        "falls back to density-only ranking.)"
    )
    if ci_st[0] > 0:
        win_ops = [op for op in ("prompt", "history", "output") if op_delta(op) > 1.0]
        lose_ops = [op for op in ("prompt", "history", "output") if op_delta(op) < -1.0]
        edge_word = "a small but" if d_st_mean < 8 else "a"
        tail = (
            f" The remaining soft spot is {', '.join(f'`{o}`' for o in lose_ops)}."
            if lose_ops
            else ""
        )
        L.append(
            f"- **Honest pitch**: bounded, *measured* token savings AND {edge_word} statistically real "
            f"key-fact-recall edge over naive truncation at equal budget — driven by "
            f"{', '.join(f'`{o}`' for o in win_ops)}.{tail}"
        )
    else:
        L.append(
            "- **Honest pitch**: sell bounded, *measured* token savings (and the structured "
            'included/excluded-section output), not "we keep more than truncation" — at a fixed budget '
            "that claim does not hold here."
        )
    L.append("")
    (RES / f"REPORT{tag}.md").write_text("\n".join(L))

    print("\n".join(L[:20]))
    print(f"\nwrote {RES / 'REPORT.md'} and {RES / 'correctness.json'}")


if __name__ == "__main__":
    main()
