"""Summarize an article with Qwen3, vanilla vs NCM-drafted speculative decoding.

Usage: uv run python examples/summarize.py [article_index]
"""

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ncm_spec  # noqa: F401,E402  (sets MLX_ENABLE_TF32 before mlx runs)
from ncm_spec.ncm_drafter import load_drafter  # noqa: E402
from ncm_spec.spec import NCMDraftAdapter, SpeculativeGenerator  # noqa: E402
from ncm_spec.target import Target  # noqa: E402


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with open(ROOT / "data" / "xsum" / "passages.pickle", "rb") as f:
        article = pickle.load(f)[idx]

    target = Target(str(ROOT / "artifacts" / "Qwen3-4B-4bit"))
    tok = target.tokenizer
    ncm = load_drafter()

    from transformers import AutoTokenizer

    gpt2 = AutoTokenizer.from_pretrained(str(ROOT / "artifacts" / "gpt2"))
    text = gpt2.decode([t for t in article if t not in (50256, 50257)])[:2000]
    content = "Summarize the following news article in one sentence.\n\n" + text
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("NCM draft :", ncm.summarize(article), "\n")
    for label, drafter in [
        ("vanilla", None),
        ("ncm-spec", NCMDraftAdapter(tok, ncm, article, max_draft_tokens=32)),
    ]:
        t0 = time.perf_counter()
        out, stats = SpeculativeGenerator(target, drafter, max_draft_tokens=32).generate(
            prompt, max_tokens=100
        )
        dt = time.perf_counter() - t0
        print(f"{label:9s}: {len(out) / dt:5.1f} tok/s | acceptance {stats.acceptance:.2f}")
        print(f"           {tok.decode(out)}\n")


if __name__ == "__main__":
    main()
