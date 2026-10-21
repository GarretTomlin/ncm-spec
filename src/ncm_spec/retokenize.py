def retokenize_draft(
    tokenizer, generated: list[int], draft_text: str, max_tokens: int
) -> list[int]:
    # joint re-encoding of the generated continuation plus the draft, with a
    # stability guard: the draft is usable only if re-encoding the generated
    # prefix is a no-op (BPE merges across the boundary would break it)
    if not generated:
        return tokenizer.encode(draft_text)[:max_tokens]
    joint = tokenizer.encode(tokenizer.decode(generated) + draft_text)
    if joint[: len(generated)] != generated:
        return []
    return joint[len(generated) : len(generated) + max_tokens]
