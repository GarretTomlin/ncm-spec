from dataclasses import dataclass

from .drafters.ngram import lookup_draft
from .retokenize import retokenize_draft


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    vanilla_rounds: int = 0

    @property
    def acceptance(self) -> float:
        return self.accepted / max(self.drafted, 1)

    @property
    def mean_accepted(self) -> float:
        n_draft_rounds = self.rounds - self.vanilla_rounds
        return self.accepted / max(n_draft_rounds, 1)


class SpeculativeGenerator:
    # greedy spec decoding: the drafter proposes target-vocab token ids, the
    # target accepts the longest exact greedy match plus one bonus token
    def __init__(self, target, drafter=None, max_draft_tokens: int = 8):
        self.target = target
        self.drafter = drafter
        self.max_draft_tokens = max_draft_tokens

    def generate(self, prompt_tokens: list[int], max_tokens: int = 128):
        self.target.reset()
        self.target.prefill(prompt_tokens)
        eos_ids = getattr(self.target.tokenizer, "eos_token_ids", None) or []
        if self.target.tokenizer.eos_token_id is not None:
            eos_ids = eos_ids + [self.target.tokenizer.eos_token_id]

        generated: list[int] = []
        stats = SpecStats()
        stop = False
        while len(generated) < max_tokens and not stop:
            stats.rounds += 1
            draft = []
            if self.drafter is not None:
                draft = self.drafter.draft(prompt_tokens, generated)[: self.max_draft_tokens]
            if draft:
                stats.drafted += len(draft)
                new, k, _ = self.target.verify(prompt_tokens[-1] if not generated else generated[-1], draft)
                stats.accepted += k
            else:
                stats.vanilla_rounds += 1
                new = [self.target.vanilla_step(generated[-1] if generated else prompt_tokens[-1])]
            for t in new:
                generated.append(t)
                if t in eos_ids:
                    stop = True
                    break
        return generated, stats


class NGramDraftAdapter:
    # prompt lookup over the full prompt + generation history
    def __init__(self, max_ngram: int = 3, num_pred: int = 10):
        self.max_ngram = max_ngram
        self.num_pred = num_pred

    def draft(self, prompt_tokens: list[int], generated: list[int]) -> list[int]:
        return lookup_draft(prompt_tokens + generated, self.max_ngram, self.num_pred)


class NCMDraftAdapter:
    # drafts a summary of a fixed article; the same draft text is re-offered
    # each round and the accepted prefix grows into it
    def __init__(self, tokenizer, ncm, article_ids, max_draft_tokens: int = 64):
        self.tokenizer = tokenizer
        self.ncm = ncm
        self.max_draft_tokens = max_draft_tokens
        self.article_ids = list(article_ids)
        self._draft_text = None

    @property
    def draft_text(self) -> str:
        if self._draft_text is None:
            self._draft_text = self.ncm.summarize(self.article_ids)
        return self._draft_text

    def draft(self, prompt_tokens: list[int], generated: list[int]) -> list[int]:
        return retokenize_draft(
            self.tokenizer, generated, self.draft_text, self.max_draft_tokens
        )


class HybridDraftAdapter:
    # n-gram when the context repeats verbatim, NCM continuation otherwise
    def __init__(self, ngram: NGramDraftAdapter, ncm: NCMDraftAdapter):
        self.ngram = ngram
        self.ncm = ncm

    def draft(self, prompt_tokens: list[int], generated: list[int]) -> list[int]:
        cand = self.ngram.draft(prompt_tokens, generated)
        return cand if cand else self.ncm.draft(prompt_tokens, generated)
