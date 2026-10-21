import mlx.core as mx


def sample(logits: mx.array, temperature: float = 0.0, top_k: int = 1) -> mx.array:
    # logits: [B, V] -> [B]; temperature 0 or top_k 1 is greedy argmax
    if temperature <= 0.0 or top_k <= 1:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    scaled = logits / temperature
    if top_k < scaled.shape[-1]:
        kth = mx.topk(scaled, top_k, axis=-1)[:, -1:]
        scaled = mx.where(scaled < kth, float("-inf"), scaled)
    probs = mx.softmax(scaled, axis=-1)
    return mx.random.categorical(probs, num_samples=1).squeeze(-1)
