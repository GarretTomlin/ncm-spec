import mlx.core as mx

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


class Target:
    # greedy verifier around an mlx-lm model; one forward pass per draft
    def __init__(self, path: str):
        self.model, self.tokenizer = load(path)
        self.cache = make_prompt_cache(self.model)

    def reset(self):
        self.cache = make_prompt_cache(self.model)

    def prefill(self, tokens: list[int]):
        self.model(mx.array([tokens], dtype=mx.uint32), cache=self.cache)
        mx.eval([c.state for c in self.cache])

    def verify(self, last_token: int, draft: list[int]) -> tuple[list[int], int, int]:
        # one pass over [last_token, *draft]; accept the longest prefix of
        # draft matching the target's greedy choices, plus one bonus token
        if not draft:
            logits = self.model(mx.array([[last_token]], dtype=mx.uint32), cache=self.cache)
            mx.eval(logits)
            return [int(mx.argmax(logits[0, -1]).item())], 0, 1
        n = len(draft)
        inp = mx.array([[last_token] + draft], dtype=mx.uint32)
        logits = self.model(inp, cache=self.cache)
        mx.eval(logits)
        greedy = mx.argmax(logits[0], axis=-1)  # [n+1]
        k = 0
        while k < n and int(greedy[k].item()) == draft[k]:
            k += 1
        bonus = int(greedy[k].item())
        trim_prompt_cache(self.cache, n - k)
        return draft[:k] + [bonus], k, n

    def vanilla_step(self, last_token: int) -> int:
        logits = self.model(mx.array([[last_token]], dtype=mx.uint32), cache=self.cache)
        mx.eval(logits)
        return int(mx.argmax(logits[0, -1]).item())

    def is_eos(self, token: int) -> bool:
        return token in getattr(self.tokenizer, "eos_token_ids", []) or token == self.tokenizer.eos_token_id
