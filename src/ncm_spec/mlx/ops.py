import math

import mlx.core as mx


def area_down_matrix(src_len: int, dst_len: int) -> mx.array:
    # F.interpolate(mode="area") = adaptive_avg_pool1d: integer bins
    # [floor(j*S/D), ceil((j+1)*S/D)) with full weight, divided by bin count
    w = mx.zeros((dst_len, src_len), dtype=mx.float32)
    for j in range(dst_len):
        s = (j * src_len) // dst_len
        e = ((j + 1) * src_len + dst_len - 1) // dst_len
        w[j, s:e] = 1.0 / (e - s)
    return w


def linear_up_matrix(src_len: int, dst_len: int) -> mx.array:
    # F.interpolate(mode="linear", align_corners=False), ATen convention:
    # src = max(0, (j+0.5)*scale - 0.5), i0 = floor(src), i1 = min(i0+1, n-1)
    scale = src_len / dst_len
    w = mx.zeros((dst_len, src_len), dtype=mx.float32)
    for j in range(dst_len):
        src = max(0.0, (j + 0.5) * scale - 0.5)
        i0 = min(math.floor(src), src_len - 1)
        i1 = min(i0 + 1, src_len - 1)
        lam1 = src - i0
        w[j, i0] += 1.0 - lam1
        w[j, i1] += lam1
    return w


def conv1d_stride(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    # Conv1d with kernel == stride, no padding: non-overlapping taps
    # x: [B, C_in, L], weight: [C_out, C_in, s], bias: [C_out]
    b, c_in, length = x.shape
    c_out, _, s = weight.shape
    xr = x.reshape(b, c_in, length // s, s)
    return mx.einsum("bins,ois->bon", xr, weight) + bias[:, None]


def conv_transpose1d_stride(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    # ConvTranspose1d with kernel == stride: out[..., j*s+t] = sum_i x[i,j] W[i,o,t]
    # x: [B, C_in, L], weight: [C_in, C_out, s], bias: [C_out]
    b, c_in, length = x.shape
    _, c_out, s = weight.shape
    out = mx.einsum("bin,iot->bont", x, weight).reshape(b, c_out, length * s)
    return out + bias[:, None]


def conv1d_k3(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    # Conv1d kernel 3, padding 1, stride 1. x: [B, C, L], weight: [C_out, C_in, 3]
    b, c, length = x.shape
    xp = mx.pad(x, [(0, 0), (0, 0), (1, 1)])
    out = None
    for k in range(3):
        term = weight[:, :, k] @ xp[:, :, k : k + length]
        out = term if out is None else out + term
    return out + bias[:, None]


def group_norm(x: mx.array, groups: int, eps: float = 1e-5) -> mx.array:
    # affine-free GroupNorm over [B, C, L]: normalize each (C/groups, L) slab
    b, c, length = x.shape
    g = x.reshape(b, groups, c // groups * length)
    mean = g.mean(axis=-1, keepdims=True)
    var = g.var(axis=-1, keepdims=True)
    g = (g - mean) * mx.rsqrt(var + eps)
    return g.reshape(b, c, length)


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array | None, eps: float = 1e-6) -> mx.array:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) * mx.rsqrt(var + eps) * weight
    return x if bias is None else x + bias


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    xf = x.astype(mx.float32)
    rms = mx.rsqrt(xf.square().mean(axis=-1, keepdims=True) + eps)
    return xf * rms * weight


def gelu(x: mx.array) -> mx.array:
    return x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))


def precompute_rope_cos_sin(positions, head_dim: int, base: float = 10000.0):
    import numpy as np

    half = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    freqs = np.asarray(positions, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
    return mx.array(cos), mx.array(sin)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    # x: [B, H, T, D], cos/sin: [T, D]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin
