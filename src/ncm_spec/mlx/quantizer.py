import mlx.core as mx
import numpy as np

from .ops import (
    area_down_matrix,
    conv1d_k3,
    conv1d_stride,
    conv_transpose1d_stride,
    group_norm,
    linear_up_matrix,
)


def _nearest_tick_idx(i: int, k: int) -> int:
    # PhiNonShared picks the phi module whose tick is closest to i/(K-1)

    if k == 4:
        ticks = np.linspace(1 / 3 / k, 1 - 1 / 3 / k, k)
    else:
        ticks = np.linspace(1 / 2 / k, 1 - 1 / 2 / k, k)
    return int(np.argmin(np.abs(ticks - i / max(k - 1, 1))))


class MultiscaleResidualVQ:
    # inference-only port of models/quant.py MultiscaleResidualVectorQuantizer
    def __init__(self, params: dict, levels: list[int], embed_dim: int = 384):
        self.levels = list(levels)
        self.K = len(levels)
        self.L = levels[-1]
        self.embed_dim = embed_dim
        self.embeddings = [params[f"vq.embeddings.{i}.weight"] for i in range(self.K)]

        self.phi_idx = [_nearest_tick_idx(i, self.K) for i in range(self.K)]
        self.phi_w = [params[f"vq.quant_resi.{i}.weight"] for i in range(self.K)]
        self.phi_b = [params[f"vq.quant_resi.{i}.bias"] for i in range(self.K)]
        self.phi_ratio = 0.5

        # cascaded learned samplers for level 0: 48 -> 12 -> 4 and back
        self.down0 = [
            (params["vq.learned_down_l1.0.weight"], params["vq.learned_down_l1.0.bias"]),
            (params["vq.learned_down_l1.2.weight"], params["vq.learned_down_l1.2.bias"]),
        ]
        self.up0 = [
            (params["vq.learned_up_l1.0.weight"], params["vq.learned_up_l1.0.bias"]),
            (params["vq.learned_up_l1.2.weight"], params["vq.learned_up_l1.2.bias"]),
        ]
        self.cascade_groups = 8

        # includes the last level: downsampling L -> L is the identity
        self._area = {lvl: area_down_matrix(self.L, lvl) for lvl in set(levels)}
        self._lin = {lvl: linear_up_matrix(lvl, self.L) for lvl in set(levels[:-1])}

    def _down_for_quant(self, z: mx.array, i: int, target_len: int) -> mx.array:
        # z: [B, L, C] -> [B, target_len, C]
        x = z.transpose(0, 2, 1)
        if i == 0:
            w, b = self.down0[0]
            x = conv1d_stride(x, w, b)
            x = group_norm(x, self.cascade_groups)
            w, b = self.down0[1]
            x = conv1d_stride(x, w, b)
            x = group_norm(x, self.cascade_groups)
        else:
            x = mx.einsum("ol,bcl->bco", self._area[target_len], x)
        return x.transpose(0, 2, 1)

    def _up_to_full(self, q: mx.array, i: int) -> mx.array:
        # q: [B, l_i, C] -> [B, L, C]
        x = q.transpose(0, 2, 1)
        if i == 0:
            w, b = self.up0[0]
            x = conv_transpose1d_stride(x, w, b)
            x = group_norm(x, self.cascade_groups)
            w, b = self.up0[1]
            x = conv_transpose1d_stride(x, w, b)
        else:
            x = mx.einsum("ol,bcl->bco", self._lin[q.shape[1]], x)
        return x.transpose(0, 2, 1)

    def _apply_resi(self, q: mx.array, i: int) -> mx.array:
        j = self.phi_idx[i]
        h = q.transpose(0, 2, 1)
        r = self.phi_ratio
        h = h * (1 - r) + conv1d_k3(h, self.phi_w[j], self.phi_b[j]) * r
        return h.transpose(0, 2, 1)

    def _nearest_codes(self, x: mx.array, i: int) -> mx.array:
        # x: [B, l, C] -> code indices [B, l] by argmin distance in fp32
        e = self.embeddings[i]
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        d = flat.square().sum(1, keepdims=True) + e.square().sum(1) - 2.0 * (flat @ e.T)
        return mx.argmin(d, axis=1).reshape(x.shape[:-1])

    def z_to_idx(self, z: mx.array) -> mx.array:
        z = z.astype(mx.float32)
        z_rest = z
        z_hat = mx.zeros_like(z)
        idx = []
        for i, level in enumerate(self.levels):
            if i != self.K - 1:
                flat = self._down_for_quant(z_rest, i, level)
            else:
                flat = z_rest
            codes = self._nearest_codes(flat, i)
            idx.append(codes)
            if i != self.K - 1:
                quantized = self._up_to_full(self.embeddings[i][codes], i)
            else:
                quantized = self.embeddings[i][codes]
            quantized = self._apply_resi(quantized, i)
            z_hat = z_hat + quantized
            z_rest = z_rest - quantized
        return mx.concatenate(idx, axis=-1)

    def idx_to_zhats(self, idx: mx.array) -> list[mx.array]:
        # cumulative z_hat after each level, at full length L
        z_hat = mx.zeros((idx.shape[0], self.L, self.embed_dim), dtype=mx.float32)
        z_hats = []
        acc = 0
        for i, level in enumerate(self.levels):
            emb = self.embeddings[i][idx[:, acc : acc + level]]
            if i != self.K - 1:
                z_q = self._up_to_full(emb, i)
            else:
                z_q = emb
            z_q = self._apply_resi(z_q, i)
            z_hat = z_hat + z_q
            z_hats.append(z_hat)
            acc += level
        return z_hats

    def idx_to_ncm_input(self, idx: mx.array) -> mx.array:
        # teacher-forced generator inputs: z_hat accumulated through levels
        # 0..K-2, area-downsampled to each next level's length; concatenated
        z_hat = mx.zeros((idx.shape[0], self.L, self.embed_dim), dtype=mx.float32)
        outs = []
        acc = 0
        for i in range(self.K - 1):
            emb = self.embeddings[i][idx[:, acc : acc + self.levels[i]]]
            z_q = self._apply_resi(self._up_to_full(emb, i), i)
            z_hat = z_hat + z_q
            nxt = mx.einsum(
                "ol,bcl->bco",
                self._area[self.levels[i + 1]],
                z_hat.transpose(0, 2, 1),
            ).transpose(0, 2, 1)
            outs.append(nxt)
            acc += self.levels[i]
        return mx.concatenate(outs, axis=1)

    def get_next_autoregressive_input(self, level_idx: int, z_hat: mx.array, z_q_level: mx.array):
        # returns updated cumulative z_hat and the input for level level_idx+1
        if level_idx != self.K - 1:
            upsampled = self._apply_resi(self._up_to_full(z_q_level, level_idx), level_idx)
            z_hat = z_hat + upsampled
            nxt = mx.einsum(
                "ol,bcl->bco",
                self._area[self.levels[level_idx + 1]],
                z_hat.transpose(0, 2, 1),
            ).transpose(0, 2, 1)
            return z_hat, nxt
        z_q_level = self._apply_resi(z_q_level, level_idx)
        return z_hat + z_q_level, z_hat + z_q_level
