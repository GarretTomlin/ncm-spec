import math

import mlx.core as mx

from .ops import gelu, layer_norm, precompute_rope_cos_sin
from .sampling import sample
from .vae import _attend, _split_qkv


class _Block:
    def __init__(self, params: dict, i: int, n_head: int):
        p = f"transformer.blocks.{i}."
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
        self.n_head = n_head

    def __call__(self, x: mx.array, mask, rope) -> mx.array:
        h = layer_norm(x, self.ln_1_w, self.ln_1_b)
        q, k, v = _split_qkv(h, self.attn_w, self.attn_b, self.n_head)
        y = _attend(q, k, v, self.n_head, self.qk_w, self.k_w, mask, rope)
        x = x + y @ self.proj_w.T + self.proj_b
        h = layer_norm(x, self.ln_2_w, self.ln_2_b)
        h = gelu(h @ self.fc_w.T + self.fc_b) @ self.mlp_proj_w.T + self.mlp_proj_b
        return x + h


class _TiedHead:
    # logits = (2 z.E^T - ||E_i||^2) / d_scale * exp(log_inv_T)
    def __init__(self, params: dict, i: int, codebook: mx.array):
        self.proj_w = params[f"lm_heads.{i}.proj.weight"]  # [384, 768]
        self.log_inv_T = float(params[f"lm_heads.{i}.log_inv_T"])
        self.e = codebook.astype(mx.float32)
        self.e_sq = self.e.square().sum(-1)
        self.d_scale = float(self.e_sq.mean())

    def __call__(self, x: mx.array) -> mx.array:
        z = x @ self.proj_w.T
        neg_d2 = (2.0 * (z @ self.e.T) - self.e_sq) / self.d_scale
        return neg_d2 * math.exp(self.log_inv_T)


class NextConceptGenerator:
    # eager generate() path for prefix-conditioned checkpoints with query_level0
    def __init__(self, params: dict, vq, cfg: dict):
        self.vq = vq
        self.levels: list[int] = cfg["levels"]
        self.K = len(self.levels)
        self.prefix_len: int = cfg["prefix_len"]
        self.n_head: int = cfg["n_head"]
        self.n_embd: int = cfg["n_embd"]
        self.vq_dim: int = vq.embed_dim
        self.pad_id: int = cfg["prefix_pad_token_id"]
        self.rope_base: float = cfg["rope_base"]
        self.encoder_wte = cfg["encoder_wte"]  # frozen VQ-VAE encoder embedding
        self.wte_w = params["transformer.wte.weight"]
        self.wte_b = params["transformer.wte.bias"]
        self.wle = params["transformer.wle.weight"]
        self.ln_f_w = params["transformer.ln_f.weight"]
        self.ln_f_b = params["transformer.ln_f.bias"]
        self.blocks = [_Block(params, i, self.n_head) for i in range(cfg["n_layer"])]
        self.heads = [_TiedHead(params, i, vq.embeddings[i]) for i in range(self.K)]
        self.l0_queries = params["l0_queries"]

        self.lvl_ids = mx.array(
            [i for i, lvl in enumerate(self.levels) for _ in range(lvl)], dtype=mx.uint32
        )
        head_dim = self.n_embd // self.n_head
        self.rope = [
            precompute_rope_cos_sin(
                list(range(self.prefix_len)) + list(range(lvl)), head_dim, self.rope_base
            )
            for lvl in self.levels
        ]
        self.masks = [self._prefix_lm_mask(lvl) for lvl in self.levels]

    def _prefix_lm_mask(self, target_len: int) -> mx.array:
        # -inf where a prefix row (i < P) looks at a target column (j >= P)
        p, t = self.prefix_len, target_len
        rows = mx.arange(p + t)[:, None]
        cols = mx.arange(p + t)[None, :]
        return mx.where((rows < p) & (cols >= p), float("-inf"), 0.0).astype(mx.float32)

    def _wte(self, z: mx.array) -> mx.array:
        return z @ self.wte_w.T + self.wte_b

    def _run_blocks(self, h: mx.array, mask: mx.array, rope) -> mx.array:
        for blk in self.blocks:
            h = blk(h, mask, rope)
        return layer_norm(h, self.ln_f_w, self.ln_f_b)

    def generate(
        self, prefix_tokens: mx.array, temperature: float = 0.0, top_k: int = 1
    ) -> mx.array:
        # prefix_tokens: [B, P] pre-token ids, right-padded with pad_id; greedy by default
        bsz = prefix_tokens.shape[0]
        prefix_tokens = prefix_tokens.astype(mx.uint32)
        prefix_emb = self._wte(self.encoder_wte[prefix_tokens])
        level_emb = self.wle[self.lvl_ids][None]

        pad_bias = mx.where(
            prefix_tokens == self.pad_id, float("-inf"), 0.0
        ).astype(mx.float32)

        def mask_for(k: int) -> mx.array:
            bias = mx.concatenate(
                [pad_bias[:, None, None, :], mx.zeros((bsz, 1, 1, self.levels[k]))],
                axis=-1,
            )
            return self.masks[k][None, None] + bias

        generated = []
        idx_next = None

        # level 0 from learned queries
        q0 = mx.broadcast_to(self.l0_queries, (bsz, self.levels[0], self.vq_dim))
        x = self._wte(q0) + self.wle[mx.zeros(self.levels[0], dtype=mx.uint32)]
        h = self._run_blocks(
            mx.concatenate([prefix_emb, x], axis=1), mask_for(0), self.rope[0]
        )
        idx_next = sample(self.heads[0](h[:, self.prefix_len :]), temperature, top_k)
        generated.append(idx_next)

        z_hat = mx.zeros((bsz, self.levels[-1], self.vq_dim), dtype=mx.float32)
        level_acum = self.levels[0]
        for level_idx in range(self.K - 1):
            z_q_level = self.vq.embeddings[level_idx][idx_next]
            z_hat, x = self.vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
            nxt = level_idx + 1
            x = self._wte(x) + level_emb[:, level_acum : level_acum + self.levels[nxt]]
            h = self._run_blocks(
                mx.concatenate([prefix_emb, x], axis=1),
                mask_for(nxt),
                self.rope[nxt],
            )
            idx_next = sample(self.heads[nxt](h[:, self.prefix_len :]), temperature, top_k)
            generated.append(idx_next)
            level_acum += self.levels[nxt]

        return mx.concatenate(generated, axis=-1)
