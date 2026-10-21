"""Convert the released torch checkpoints to safetensors for the MLX runtime.

Run: uv run python -m ncm_spec.convert artifacts/xsum
Writes: <dir>/tokenizer.mlx.safetensors, <dir>/generator.mlx.safetensors,
        <dir>/token_unigram.npy
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

SKIP_TOKENIZER = ("vq.ema_", "vq.codebook_hit_")
SKIP_GENERATOR = ("lvl_1L", "attn_mask_for_masking")


def convert_dir(ckpt_dir: Path) -> None:
    tok = torch.load(ckpt_dir / "tokenizer.pt", map_location="cpu", weights_only=False)["model"]
    out = {
        k: v.numpy().astype(np.float32)
        for k, v in tok.items()
        if not k.startswith(SKIP_TOKENIZER)
    }
    save_file(out, str(ckpt_dir / "tokenizer.mlx.safetensors"))
    print(f"tokenizer: {len(out)} tensors")

    gen_ckpt = torch.load(ckpt_dir / "generator.pt", map_location="cpu", weights_only=False)
    gen = {
        k: v.numpy().astype(np.float32)
        for k, v in gen_ckpt["model"].items()
        if k not in SKIP_GENERATOR
    }
    save_file(gen, str(ckpt_dir / "generator.mlx.safetensors"))
    print(f"generator: {len(gen)} tensors")

    unigram = gen_ckpt["inference"]["token_unigram"].numpy().astype(np.float64)
    np.save(ckpt_dir / "token_unigram.npy", unigram)
    print(f"token_unigram: {unigram.shape}")


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    convert_dir(Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
