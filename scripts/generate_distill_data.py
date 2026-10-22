"""Generate Qwen3 greedy summaries of XSum articles for drafter distillation.

The article text is exactly what the NCM drafter sees at inference: GPT-2
decode of the first 512 article token ids. Summaries are greedy, thinking
disabled, capped at 64 tokens. Articles 0-999 are held out so benchmarks
stay clean. Resumable via JSONL append.

Usage: uv run python scripts/generate_distill_data.py
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ncm_spec  # noqa: F401,E402,I001  (sets MLX_ENABLE_TF32 before mlx runs)
from ncm_spec.ncm_drafter import load_drafter  # noqa: E402
from ncm_spec.spec import SpeculativeGenerator  # noqa: E402
from ncm_spec.target import Target  # noqa: E402

INSTR = "Summarize the following news article in one sentence.\n\n"
HOLDOUT = 1000  # articles before this index are never trained on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default=ROOT / "data" / "distill_summaries.jsonl")
    a = ap.parse_args()

    ncm = load_drafter()
    gpt2 = ncm.pre_tokenizer

    with open(ROOT / "data" / "xsum" / "passages.pickle", "rb") as f:
        passages = pickle.load(f)
    todo = list(range(HOLDOUT, len(passages)))

    target = Target(str(ROOT / "artifacts" / "Qwen3-4B-4bit"))
    tok = target.tokenizer
    eos = set(getattr(tok, "eos_token_ids", None) or [])
    eos.add(tok.eos_token_id)

    done = set()
    if a.out.exists():
        with open(a.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["article_idx"])
                except json.JSONDecodeError:
                    continue
    todo = [i for i in todo if i not in done]
    print(f"{len(done)} already generated, {len(todo)} to go", flush=True)

    gen = SpeculativeGenerator(target)
    t0 = time.perf_counter()
    count = 0
    with open(a.out, "a") as fout:
        for idx in todo:
            ids = passages[idx]
            text = gpt2.decode([t for t in ids[:512] if t not in (50256, 50257)])
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": INSTR + text}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            out, _ = gen.generate(prompt, max_tokens=a.max_tokens)
            summary = tok.decode([t for t in out if t not in eos])
            fout.write(json.dumps({"article_idx": idx, "summary": summary}) + "\n")
            fout.flush()
            count += 1
            if count % 50 == 0:
                rate = count / (time.perf_counter() - t0)
                eta = (len(todo) - count) / max(rate, 1e-9) / 3600
                print(f"{count}/{len(todo)} ({rate * 3600:.0f}/h, eta {eta:.1f}h)", flush=True)
    print("done", count + len(done), flush=True)


if __name__ == "__main__":
    main()
