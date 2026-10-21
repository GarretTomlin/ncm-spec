import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

XSUM_CKPT = ["config.yaml", "generator.pt", "tokenizer.pt", "README.md"]
GPT2_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def main():
    snapshot_download(
        "hazyresearch/ncm-checkpoints",
        local_dir=ROOT / "artifacts" / "xsum",
        allow_patterns=[f"xsum/{f}" for f in XSUM_CKPT],
    )
    snapshot_download(
        "hazyresearch/ncm-datasets",
        repo_type="dataset",
        local_dir=ROOT / "data",
        allow_patterns=["xsum/*"],
    )
    snapshot_download(
        "openai-community/gpt2",
        local_dir=ROOT / "artifacts" / "gpt2",
        allow_patterns=GPT2_FILES,
    )
    snapshot_download(
        "mlx-community/Qwen3-4B-4bit",
        local_dir=ROOT / "artifacts" / "Qwen3-4B-4bit",
    )
    for p in sorted((ROOT / "artifacts").rglob("*")):
        if p.is_file():
            print(p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    sys.exit(main())
