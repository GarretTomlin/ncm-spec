# ncm-spec benchmarks

target: Qwen3-4B-4bit (mlx-community) | articles: 20 | max tokens: 100 | device: Apple Apple M5

## summarize

| method | tok/s (mean) | speedup | acceptance | mean accepted | identical |
|---|---|---|---|---|---|
| vanilla | 39.4 | 1.00x | 0.00 | 0.0 | n/a |
| ngram | 28.4 | 0.72x | 0.06 | 0.6 | 20/20 |
| ncm | 13.4 | 0.34x | 0.00 | 0.0 | 17/20 |
| hybrid | 18.6 | 0.47x | 0.04 | 0.5 | 18/20 |

## quote

| method | tok/s (mean) | speedup | acceptance | mean accepted | identical |
|---|---|---|---|---|---|
| vanilla | 41.1 | 1.00x | 0.00 | 0.0 | n/a |
| ngram | 106.0 | 2.58x | 0.67 | 6.3 | 20/20 |
| hybrid | 43.6 | 1.06x | 0.53 | 5.3 | 20/20 |
