"""Fine-tune the NCM generator on target-model summaries (drafter distillation).

Teacher codes come from the frozen VQ-VAE encode of each summary; the loss is
cross-entropy over all level codes in one teacher-forced pass. The tokenizer
stays frozen; only generator weights update.

Usage:
  uv run python -m ncm_spec.train --data data/distill_summaries.jsonl \
      --steps 3000 --batch 16 --lr 2e-5 --out artifacts/xsum_distilled
"""

import os

# fine-tuning tolerates TF32 (the checkpoint was originally trained in bf16);
# must be set before ncm_spec applies its inference default
os.environ["MLX_ENABLE_TF32"] = "1"

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ncm_spec  # noqa: F401,E402

from .mlx.generator import NextConceptGenerator
from .ncm_drafter import ROOT, load_drafter

HOLDOUT = 1000


def load_examples(data_path: Path, passages, drafter):
    gpt2 = drafter.pre_tokenizer
    vq = drafter.tokenizer.vq
    prefix_len = drafter.prefix_len
    pad_id = drafter.pad_id
    tok = drafter.tokenizer

    examples = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            idx = row["article_idx"]
            art = list(passages[idx])[:prefix_len]
            prefix = art + [pad_id] * (prefix_len - len(art))
            ids = gpt2.encode(row["summary"])[: tok.context_length]
            if not ids:
                continue
            ids = ids + [pad_id] * (tok.context_length - len(ids))
            codes = tok.encode(mx.array([ids], dtype=mx.uint32))
            examples.append((prefix, codes[0].tolist(), idx))
    return examples


def collate(batch, vq):
    prefix = mx.array([b[0] for b in batch], dtype=mx.uint32)
    codes = mx.array([b[1] for b in batch], dtype=mx.uint32)
    ncm_input = vq.idx_to_ncm_input(codes)
    return prefix, codes, ncm_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "distill_summaries.jsonl")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "xsum_distilled")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    drafter = load_drafter()
    vq = drafter.tokenizer.vq
    cfg = json.loads((ROOT / "configs" / "xsum_drafter.json").read_text())
    cfg["encoder_wte"] = drafter.tokenizer.wte

    with open(ROOT / "data" / "xsum" / "passages.pickle", "rb") as f:
        passages = pickle.load(f)

    print(f"loading examples from {a.data}", flush=True)
    examples = load_examples(a.data, passages, drafter)
    n_val = max(8, int(len(examples) * a.val_frac))
    train, val = examples[:-n_val], examples[-n_val:]
    print(f"{len(train)} train / {n_val} val", flush=True)
    val_batch = collate(val[:8], vq)

    params = mx.load(str(ROOT / "artifacts" / "xsum" / "generator.mlx.safetensors"))
    gen = NextConceptGenerator(params, vq, cfg)
    levels = gen.levels
    total_codes = sum(levels)

    model = nn.Module()
    model.params = params

    def loss_fn(m, prefix, codes, ncm_input):
        g = NextConceptGenerator(m.params, vq, cfg)
        logits = g.forward_train(prefix, ncm_input)
        nll = 0.0
        off = 0
        for k, lvl in enumerate(levels):
            lg = logits[k].reshape(-1, logits[k].shape[-1])
            tg = codes[:, off : off + lvl].reshape(-1)
            nll = nll + mx.sum(nn.losses.cross_entropy(lg, tg))
            off += lvl
        return nll / (prefix.shape[0] * total_codes)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    optimizer = optim.AdamW(learning_rate=a.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        t = (step - a.warmup) / max(a.steps - a.warmup, 1)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    def teacher_match(m, batch):
        prefix, codes, ncm_input = batch
        g = NextConceptGenerator(m.params, vq, cfg)
        logits = g.forward_train(prefix, ncm_input)
        agree, off = 0, 0
        for k, lvl in enumerate(levels):
            pred = mx.argmax(logits[k], axis=-1)
            agree = agree + mx.sum(pred == codes[:, off : off + lvl])
            off += lvl
        return agree / (prefix.shape[0] * total_codes)

    print("initial val loss", float(loss_fn(model, *val_batch)), flush=True)

    order = list(range(len(train)))
    t0 = time.perf_counter()
    for step in range(a.steps):
        if step * a.batch % len(order) == 0:
            random.shuffle(order)
        batch_idx = [
            order[(step * a.batch + j) % len(order)] for j in range(a.batch)
        ]
        batch = collate([train[i] for i in batch_idx], vq)
        optimizer.learning_rate = lr_at(step)
        loss, grads = loss_and_grad(model, *batch)
        grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.params, optimizer.state)
        if step % 25 == 0 or step == a.steps - 1:
            dt = time.perf_counter() - t0
            print(f"step {step}: loss {float(loss):.4f} gnorm {float(grad_norm):.2f} ({dt:.0f}s)", flush=True)
        if step % 250 == 0 and step > 0:
            m = float(teacher_match(model, val_batch))
            print(f"  val teacher-code match {m:.3f}", flush=True)

    params = model.params
    a.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(a.out / "generator.safetensors"), params)
    # tokenizer and decode priors are unchanged
    import shutil

    for name in ("tokenizer.mlx.safetensors", "token_unigram.npy"):
        src = ROOT / "artifacts" / "xsum" / name
        if not (a.out / name).exists():
            shutil.copy(src, a.out / name)
    (a.out / "config.json").write_text(
        (ROOT / "configs" / "xsum_drafter.json").read_text()
    )
    print(f"saved to {a.out}", flush=True)
    print("final val loss", float(loss_fn(model, *val_batch)), flush=True)
    print("final teacher-code match", float(teacher_match(model, val_batch)), flush=True)


if __name__ == "__main__":
    main()
