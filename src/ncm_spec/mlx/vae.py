import mlx.core as mx

from .ops import gelu, group_norm, layer_norm, rms_norm


def _split_qkv(x: mx.array, w: mx.array, b: mx.array, n_head: int):
    # x: [B, T, C] -> q, k, v each [B, H, T, C//H]
    bsz, t, c = x.shape
    qkv = x @ w.T + b
    q, k, v = mx.split(qkv, 3, axis=-1)
    hd = c // n_head
    q = q.reshape(bsz, t, n_head, hd).transpose(0, 2, 1, 3)
    k = k.reshape(bsz, t, n_head, hd).transpose(0, 2, 1, 3)
    v = v.reshape(bsz, t, n_head, hd).transpose(0, 2, 1, 3)
    return q, k, v


def _attend(q, k, v, n_head: int, qk_w, k_w, mask, rope):
    from .ops import apply_rope

    q = rms_norm(q, qk_w)
    k = rms_norm(k, k_w)
    if rope is not None:
        cos, sin = rope
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    y = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=1.0 / (q.shape[-1] ** 0.5), mask=mask
    )
    return y.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)


class Decoder:
    # single GPT-2 style block over the full sequence, bidirectional attention
    def __init__(self, params: dict, embed_dim: int = 384, n_head: int = 6):
        p = "decoder.middle_block."
        self.ln_1_w, self.ln_1_b = params[p + "ln_1.weight"], params[p + "ln_1.bias"]
        self.attn_w = params[p + "attn.c_attn.weight"]
        self.attn_b = params[p + "attn.c_attn.bias"]
        self.proj_w = params[p + "attn.c_proj.weight"]
        self.proj_b = params[p + "attn.c_proj.bias"]
        self.qk_w = params[p + "attn.q_norm.weight"]
        self.k_w = params[p + "attn.k_norm.weight"]
        self.ln_2_w, self.ln_2_b = params[p + "ln_2.weight"], params[p + "ln_2.bias"]
        self.fc_w = params[p + "mlp.c_fc.weight"]
        self.fc_b = params[p + "mlp.c_fc.bias"]
        self.mlp_proj_w = params[p + "mlp.c_proj.weight"]
        self.mlp_proj_b = params[p + "mlp.c_proj.bias"]
        self.ln_f_w = params["decoder.ln_f.weight"]
        self.ln_f_b = params.get("decoder.ln_f.bias")
        self.out_w = params["decoder.output_layer.weight"]
        self.n_head = n_head

    def __call__(self, z: mx.array) -> mx.array:
        x = layer_norm(z, self.ln_1_w, self.ln_1_b)
        q, k, v = _split_qkv(x, self.attn_w, self.attn_b, self.n_head)
        y = _attend(q, k, v, self.n_head, self.qk_w, self.k_w, None, None)
        x = z + y @ self.proj_w.T + self.proj_b
        h = layer_norm(x, self.ln_2_w, self.ln_2_b)
        h = gelu(h @ self.fc_w.T + self.fc_b) @ self.mlp_proj_w.T + self.mlp_proj_b
        x = x + h
        x = layer_norm(x, self.ln_f_w, self.ln_f_b)
        return x @ self.out_w.T


class VQVAETokenizer:
    def __init__(self, params: dict, levels: list[int], embed_dim: int = 384):
        from .quantizer import MultiscaleResidualVQ

        self.wte = params["encoder.wte.weight"]
        self.wpe = params["encoder.wpe.weight"]
        self.pre_quant_groups = 8
        self.vq = MultiscaleResidualVQ(params, levels, embed_dim)
        self.decoder = Decoder(params, embed_dim, n_head=6)
        self.context_length = levels[-1]

    def encode_pre_quant(self, idx: mx.array) -> mx.array:
        # idx: [B, L] -> z: [B, L, C]
        z = self.wte[idx] + self.wpe[: idx.shape[1]]
        z = group_norm(z.transpose(0, 2, 1), self.pre_quant_groups).transpose(0, 2, 1)
        return z

    def encode(self, idx: mx.array) -> mx.array:
        return self.vq.z_to_idx(self.encode_pre_quant(idx))

    def decode_codes(self, codes: mx.array) -> mx.array:
        # codes: [B, sum(levels)] -> token logits [B, L, vocab]
        z_hat = self.vq.idx_to_zhats(codes)[-1]
        return self.decoder(z_hat)

    def reconstruct(self, idx: mx.array) -> mx.array:
        codes = self.encode(idx)
        return self.decode_codes(codes)
