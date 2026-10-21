import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .mlx.generator import NextConceptGenerator
from .mlx.vae import VQVAETokenizer

ROOT = Path(__file__).resolve().parent.parent.parent


def load_drafter(ckpt_dir: str | Path = None, config: str | Path = None):
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else ROOT / "artifacts" / "xsum"
    config = Path(config) if config else ROOT / "configs" / "xsum_drafter.json"
    cfg = json.loads(config.read_text())

    tok_params = mx.load(str(ckpt_dir / "tokenizer.mlx.safetensors"))
    gen_file = "generator.safetensors"
    if not (ckpt_dir / gen_file).exists():
        gen_file = "generator.mlx.safetensors"
    gen_params = mx.load(str(ckpt_dir / gen_file))
    unigram = np.load(ckpt_dir / "token_unigram.npy")

    tokenizer = VQVAETokenizer(tok_params, cfg["levels"], cfg["embed_dim"])
    gcfg = {
        **cfg,
        "encoder_wte": tokenizer.wte,
    }
    generator = NextConceptGenerator(gen_params, tokenizer.vq, gcfg)
    return NCMDrafter(tokenizer, generator, cfg, unigram)


class NCMDrafter:
    def __init__(self, tokenizer, generator, cfg, unigram):
        self.tokenizer = tokenizer
        self.generator = generator
        self.cfg = cfg
        self.prefix_len = cfg["prefix_len"]
        self.pad_id = cfg["prefix_pad_token_id"]
        self.eos_id = cfg["eos_token_id"]
        self.unigram = unigram
        d = cfg["decoder"]
        self.freq = d["frequency_penalty"]
        self.mu = d["frequency_deficit"]
        self.adj = tuple(d["adjacency_penalty"])
        self._pre_tok = None

    @property
    def pre_tokenizer(self):
        # GPT-2 with <|pad|> as id 50257, matching the released checkpoint
        if self._pre_tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.cfg["pre_tokenizer"])
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            self._pre_tok = tok
        return self._pre_tok

    def build_prefix(self, article: str | list[int]) -> mx.array:
        if isinstance(article, str):
            ids = self.pre_tokenizer.encode(article)
        else:
            ids = list(article)
        ids = ids[: self.prefix_len]
        ids = ids + [self.pad_id] * (self.prefix_len - len(ids))
        return mx.array([ids], dtype=mx.uint32)

    def _pick_tokens(self, logits: np.ndarray) -> np.ndarray:
        # greedy argmax with the released frequency/adjacency penalties, fp64
        if self.freq <= 0 and not self.adj:
            return logits.argmax(-1)
        bsz, seq, vocab = logits.shape
        lg = logits.astype(np.float64)
        allow = self.unigram[:vocab] * seq
        al, aw, ad = self.adj[0], int(self.adj[1]), self.adj[2]
        cnt = np.zeros((bsz, vocab))
        out = np.empty((bsz, seq), dtype=np.int64)
        hist = []
        for t in range(seq):
            step = lg[:, t, :].copy()
            if self.freq > 0:
                step -= self.freq * np.maximum(cnt - allow, 0.0)
                if self.mu > 0:
                    step += self.mu * ((seq - t) / seq) * np.maximum(allow - cnt, 0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step[np.arange(bsz), h] -= al * ad**j
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt[np.arange(bsz), p] += 1.0
        return out

    def summarize(self, article: str) -> str:
        prefix = self.build_prefix(article)
        codes = self.generator.generate(prefix)
        logits = np.array(self.tokenizer.decode_codes(codes))
        ids = self._pick_tokens(logits)[0].tolist()
        if self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        ids = [t for t in ids if t not in (self.eos_id, self.pad_id)]
        return self.pre_tokenizer.decode(ids, skip_special_tokens=True)
