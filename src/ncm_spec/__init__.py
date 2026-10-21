import os

# TF32 fp32 matmuls break bit-parity with the reference implementation and can
# flip nearest-code assignments; disable unless the caller opts out.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
