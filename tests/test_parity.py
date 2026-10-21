import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "xsum"

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

sys.path.insert(0, str(ROOT / "vendor" / "ncm"))


@pytest.fixture(scope="module")
def reference():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(ART / "config.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = str(ART / "tokenizer.pt")
    from tokenization.vqvae_tokenizer import VQVAETokenizer

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae.eval()
    from models.next_concept import NextConceptTransformer

    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model)
    net.load_state_dict(
        torch.load(ART / "generator.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    net.eval()
    return tk, net


@pytest.fixture(scope="module")
def ours():
    from ncm_spec.ncm_drafter import load_drafter

    return load_drafter()


@pytest.fixture(scope="module")
def articles():
    with open(ROOT / "data" / "xsum" / "passages.pickle", "rb") as f:
        return pickle.load(f)[:8]


def _pad48(ids, pad=50257):
    ids = ids[:48]
    return ids + [pad] * (48 - len(ids))


def test_vqvae_encode_parity(reference, ours, articles):
    tk, _ = reference
    agree, total = 0, 0
    for ids in articles:
        ids = _pad48(ids)
        ref = tk.vqvae.encode(torch.tensor([ids])).view(-1).tolist()
        import mlx.core as mx

        got = mx.array([ids], dtype=mx.uint32)
        out = ours.tokenizer.encode(got).tolist()[0]
        agree += sum(a == b for a, b in zip(ref, out))
        total += len(ref)
    assert agree / total >= 0.999, f"quantizer code agreement {agree}/{total}"


def test_vqvae_decode_parity(reference, ours, articles):
    tk, _ = reference
    agree, total = 0, 0
    for ids in articles:
        ids = _pad48(ids)
        with torch.no_grad():
            # eval_shard.py decode path: encode -> idx_to_zhats -> decoder argmax
            codes = tk.vqvae.encode(torch.tensor([ids]))
            _, zl = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
            ref = tk.vqvae.decoder(zl[-1]).argmax(-1).view(-1).tolist()
        import mlx.core as mx

        got = ours.tokenizer.reconstruct(mx.array([ids], dtype=mx.uint32))
        got = got.argmax(-1).tolist()[0]
        agree += sum(a == b for a, b in zip(ref, got))
        total += len(ref)
    assert agree / total >= 0.999, f"reconstruction agreement {agree}/{total}"


def test_generator_parity(reference, ours, articles):
    _, net = reference
    agree, total = 0, 0
    for ids in articles:
        prefix = ours.build_prefix(ids)
        with torch.no_grad():
            ref = (
                net.generate(
                    None,
                    torch.tensor(prefix.tolist()),
                    temperature=1.0,
                    top_k=1,
                    top_p=0.0,
                )
                .view(-1)
                .tolist()
            )
        got = ours.generator.generate(prefix).tolist()[0]
        same = sum(a == b for a, b in zip(ref, got))
        agree += same
        total += len(ref)
    assert agree / total >= 0.95, f"generator code agreement {agree}/{total}"


def test_summarize(reference, ours, articles):
    tk, net = reference

    def torch_summary(ids):
        ids = list(ids)[:512]
        ids = ids + [50257] * (512 - len(ids))
        with torch.no_grad():
            g = net.generate(
                None, torch.tensor([ids]), temperature=1.0, top_k=1, top_p=0.0
            )
            _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
            logits = tk.vqvae.decoder(zl[-1])
        ref_ids = pick_tokens(
            logits,
            freq=2.0,
            mu=1.0,
            adj=(5.0, 4, 0.5),
            allow=ours.unigram,
        )[0].tolist()
        if 50256 in ref_ids:
            ref_ids = ref_ids[: ref_ids.index(50256)]
        ref_ids = [t for t in ref_ids if t not in (50256, 50257)]
        return ours.pre_tokenizer.decode(ref_ids, skip_special_tokens=True)

    for ids in articles[:4]:
        assert ours.summarize(ids) == torch_summary(ids)


def pick_tokens(logits, freq, mu, adj, allow):
    # eval_shard.py pick_tokens, fp64 greedy with running penalties
    bsz, seq, vocab = logits.shape
    lg = logits.double()
    allow = torch.as_tensor(allow, dtype=torch.float64)
    allow = (allow[:vocab] * seq)[None, :]
    al, aw, ad = adj
    cnt = torch.zeros(bsz, vocab, dtype=torch.float64)
    out = torch.empty(bsz, seq, dtype=torch.long)
    hist = []
    for t in range(seq):
        step = lg[:, t, :].clone()
        if freq > 0:
            step -= freq * (cnt - allow).clamp_min(0.0)
            if mu > 0:
                step += mu * ((seq - t) / seq) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((bsz, 1), -al * ad**j, dtype=torch.float64))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(bsz, 1, dtype=torch.float64))
    return out
