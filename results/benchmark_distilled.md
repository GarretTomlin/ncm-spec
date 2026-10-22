# ncm-spec benchmarks

target: Qwen3-4B-4bit (mlx-community) | articles: 20 | max tokens: 100 | device: Apple Apple M5

## summarize

| method | tok/s (mean) | speedup | acceptance | mean accepted | identical |
|---|---|---|---|---|---|
| vanilla | 39.0 | 1.00x | 0.00 | 0.0 | n/a |
| ngram | 28.7 | 0.74x | 0.06 | 0.6 | 20/20 |
| ncm | 16.1 | 0.41x | 0.00 | 0.0 | 17/20 |
| hybrid | 20.8 | 0.53x | 0.06 | 0.5 | 19/20 |

## quote

| method | tok/s (mean) | speedup | acceptance | mean accepted | identical |
|---|---|---|---|---|---|
| vanilla | 38.8 | 1.00x | 0.00 | 0.0 | n/a |
| ngram | 121.1 | 3.12x | 0.67 | 6.3 | 20/20 |
| hybrid | 44.8 | 1.15x | 0.59 | 4.1 | 20/20 |
