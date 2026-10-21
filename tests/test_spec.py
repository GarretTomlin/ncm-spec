from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HAS_ARTIFACTS = (ROOT / "artifacts" / "Qwen3-4B-4bit").exists() and (
    ROOT / "artifacts" / "xsum" / "tokenizer.mlx.safetensors"
).exists()


def test_lookup_draft_finds_continuation():
    from ncm_spec.drafters.ngram import lookup_draft

    hist = [5, 6, 7, 8, 9, 5, 6, 7, 1, 2, 5, 6, 7]
    assert lookup_draft(hist, max_ngram=3, num_pred=4) == [1, 2, 5, 6]


def test_lookup_draft_no_match():
    from ncm_spec.drafters.ngram import lookup_draft

    assert lookup_draft([1, 2, 3], max_ngram=3, num_pred=4) == []


def test_retokenize_guard():
    from ncm_spec.retokenize import retokenize_draft

    class Tok:
        def encode(self, s):
            if s == "10|x":  # emulates a BPE merge across the boundary
                return [100]
            return [int(w) for w in s.split("|")]

        def decode(self, ids):
            return "|".join(str(i) for i in ids) + "|"

    # stable prefix: draft tokens are the tail
    assert retokenize_draft(Tok(), [10, 11], "20|21", 8) == [20, 21]
    # boundary merge changes the prefix: rejected
    assert retokenize_draft(Tok(), [10], "x", 8) == []
    # nothing generated yet: draft encodes directly
    assert retokenize_draft(Tok(), [], "20|21", 8) == [20, 21]


@pytest.mark.skipif(not HAS_ARTIFACTS, reason="artifacts not downloaded")
def test_spec_matches_vanilla_greedy():
    import pickle

    from ncm_spec.spec import NGramDraftAdapter, SpeculativeGenerator
    from ncm_spec.target import Target

    target = Target(str(ROOT / "artifacts" / "Qwen3-4B-4bit"))
    tok = target.tokenizer
    with open(ROOT / "data" / "xsum" / "passages.pickle", "rb") as f:
        article = pickle.load(f)[0]
    text = tok.decode([t for t in article if t < 151000])[:1500]
    content = "Repeat the first sentence of this article exactly, then say DONE.\n\n" + text
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    vanilla, _ = SpeculativeGenerator(target).generate(prompt, max_tokens=80)
    spec, stats = SpeculativeGenerator(
        target, NGramDraftAdapter(num_pred=12), max_draft_tokens=12
    ).generate(prompt, max_tokens=80)
    assert spec == vanilla
    assert stats.acceptance > 0
