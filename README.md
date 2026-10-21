# ncm-spec

Concept-level speculative decoding for local LLMs on Apple Silicon.

A small [NCM](https://arxiv.org/abs/2610.14162) (Next-Concept Modeling) model drafts
text coarse-to-fine in parallel; a target model (Qwen3 via [mlx-lm](https://github.com/ml-explore/mlx-lm))
verifies each draft in a single forward pass. Work in progress.
