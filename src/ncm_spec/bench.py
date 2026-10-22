"""Benchmark runner: vanilla vs n-gram vs NCM vs hybrid speculative decoding.

Usage: uv run python -m ncm_spec.bench [--articles 20] [--out results]
"""

import argparse
import json
import pickle
import subprocess
import time
from pathlib import Path

from .ncm_drafter import ROOT, load_drafter
from .spec import (
    HybridDraftAdapter,
    NCMDraftAdapter,
    NGramDraftAdapter,
    SpeculativeGenerator,
)
from .target import Target

SUMMARIZE_INSTR = "Summarize the following news article in one sentence.\n\n"
QUOTE_INSTR = "Repeat the first sentence of this article exactly, then say DONE.\n\n"

TASKS = {
    "summarize": (SUMMARIZE_INSTR, ["vanilla", "ngram", "ncm", "hybrid"]),
    "quote": (QUOTE_INSTR, ["vanilla", "ngram", "hybrid"]),
}


def build_drafter(name, target_tok, ncm, article_ids, ncm_draft_tokens):
    if name == "ngram":
        return NGramDraftAdapter(num_pred=12)
    if name == "ncm":
        return NCMDraftAdapter(target_tok, ncm, article_ids, max_draft_tokens=ncm_draft_tokens)
    if name == "hybrid":
        return HybridDraftAdapter(
            NGramDraftAdapter(num_pred=12),
            NCMDraftAdapter(target_tok, ncm, article_ids, max_draft_tokens=ncm_draft_tokens),
        )
    return None


def run_bench(articles, max_tokens=100, warmup=2, target_path=None,
              ncm_artifacts=None, ncm_draft_tokens=8):
    target = Target(target_path or str(ROOT / "artifacts" / "Qwen3-4B-4bit"))
    tok = target.tokenizer
    ncm = load_drafter(ncm_artifacts) if ncm_artifacts else load_drafter()
    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained(str(ROOT / "artifacts" / "gpt2"))

    texts = []
    for ids in articles:
        clean = [t for t in ids if t not in (50256, 50257)]
        texts.append(gpt2.decode(clean)[:2000])

    results = {}
    for task, (instr, methods) in TASKS.items():
        results[task] = {m: {"tok_s": [], "acceptance": [], "mean_accepted": [], "identical": []}
                         for m in methods}
        for n, (ids, text) in enumerate(zip(articles, texts)):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": instr + text}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            vanilla_out = None
            for method in methods:
                target.reset()
                target.prefill(prompt)
                drafter = build_drafter(method, tok, ncm, ids, ncm_draft_tokens)
                cap = min(12, ncm_draft_tokens) if method in ("ncm", "hybrid") else 12
                gen = SpeculativeGenerator(target, drafter, max_draft_tokens=cap)
                if n < warmup:
                    gen.generate(prompt, max_tokens, prepare=False)
                    target.reset()
                    target.prefill(prompt)
                t0 = time.perf_counter()
                out, stats = gen.generate(prompt, max_tokens, prepare=False)
                dt = time.perf_counter() - t0
                row = results[task][method]
                row["tok_s"].append(len(out) / dt)
                row["acceptance"].append(stats.acceptance)
                row["mean_accepted"].append(stats.mean_accepted)
                if method == "vanilla":
                    vanilla_out = out
                else:
                    row["identical"].append(out == vanilla_out)
        print(f"[{task}] done")
    return results


def to_markdown(results, meta):
    lines = [
        "# ncm-spec benchmarks",
        "",
        f"target: {meta['target']} | articles: {meta['articles']} | "
        f"max tokens: {meta['max_tokens']} | device: Apple {meta['chip']}",
        "",
    ]
    for task, methods in results.items():
        header = "| method | tok/s (mean) | speedup | acceptance | mean accepted | identical |"
        lines += [f"## {task}", "", header, "|---|---|---|---|---|---|"]
        base = sum(methods["vanilla"]["tok_s"]) / len(methods["vanilla"]["tok_s"])
        for name, row in methods.items():
            tps = sum(row["tok_s"]) / len(row["tok_s"])
            acc = sum(row["acceptance"]) / len(row["acceptance"])
            ma = sum(row["mean_accepted"]) / len(row["mean_accepted"])
            if name == "vanilla":
                ident = "n/a"
            else:
                ident = f"{sum(row['identical'])}/{len(row['identical'])}"
            lines.append(
                f"| {name} | {tps:.1f} | {tps / base:.2f}x | {acc:.2f} | {ma:.1f} | {ident} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ncm-artifacts", default=None, help="distilled drafter checkpoint dir")
    ap.add_argument("--ncm-draft-tokens", type=int, default=8)
    a = ap.parse_args()

    with open(ROOT / "data" / "xsum" / "passages.pickle", "rb") as f:
        articles = pickle.load(f)[: a.articles]

    results = run_bench(
        articles, a.max_tokens, ncm_artifacts=a.ncm_artifacts,
        ncm_draft_tokens=a.ncm_draft_tokens,
    )
    meta = {
        "target": "Qwen3-4B-4bit (mlx-community)",
        "articles": a.articles,
        "max_tokens": a.max_tokens,
        "ncm_artifacts": a.ncm_artifacts or "artifacts/xsum (original)",
        "ncm_draft_tokens": a.ncm_draft_tokens,
        "chip": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
    }
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    suffix = "_distilled" if a.ncm_artifacts else ""
    (out / f"benchmark{suffix}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2)
    )
    md = to_markdown(results, meta)
    (out / f"benchmark{suffix}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
