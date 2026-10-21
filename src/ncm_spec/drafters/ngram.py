def lookup_draft(history: list[int], max_ngram: int = 3, num_pred: int = 8) -> list[int]:
    # prompt lookup decoding: extend the longest n-gram match of the trailing
    # context with the tokens that followed it earlier in the history
    for n in range(min(max_ngram, len(history) - 1), 0, -1):
        needle = tuple(history[-n:])
        for i in range(len(history) - n - 1, -1, -1):
            if tuple(history[i : i + n]) == needle:
                cand = history[i + n : i + n + num_pred]
                if cand:
                    return cand
    return []


class NGramDrafter:
    def __init__(self, max_ngram: int = 3, num_pred: int = 8):
        self.max_ngram = max_ngram
        self.num_pred = num_pred

    def draft(self, tokens: list[int]) -> list[int]:
        return lookup_draft(tokens, self.max_ngram, self.num_pred)
