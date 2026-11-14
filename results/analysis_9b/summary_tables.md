# Summary tables (Qwen3.5-9B)

Recall@10 in %, macro over five tasks, item-weighted. Distances are means over the seven layers.

## Recall@10 by prompts, Readout Filtering on every lens (mean of pools A and B)

| dataset | lens | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|
| en | j | 37.6 | 37.6 | 37.5 | 37.7 |
| en | jpp | 57.1 | 55.1 | 57.9 | 57.9 |
| en | logit | 32.8 | 32.8 | 32.8 | 32.8 |
| en | r | 38.3 | 38.4 | 37.3 | 37.7 |
| fr | j | 29.6 | 29.5 | 29.6 | 30.5 |
| fr | jpp | 48.0 | 48.7 | 48.9 | 48.2 |
| fr | logit | 32.8 | 32.8 | 32.8 | 32.8 |
| fr | r | 30.7 | 30.8 | 30.4 | 30.6 |
| wikitext | j | 38.9 | 38.3 | 38.4 | 38.0 |
| wikitext | jpp | 55.5 | 55.3 | 56.1 | 56.5 |
| wikitext | logit | 32.8 | 32.8 | 32.8 | 32.8 |
| wikitext | r | 43.3 | 43.7 | 43.5 | 43.2 |

## Recall@10 at 1000 prompts, pool A vs pool B (filtered)

| dataset | lens | a | b |
|---|---|---|---|
| en | j | 37.9 | 37.6 |
| en | jpp | 57.1 | 58.7 |
| en | logit | 32.8 | 32.8 |
| en | r | 37.7 | 37.7 |
| fr | j | 30.5 | 30.5 |
| fr | jpp | 48.7 | 47.7 |
| fr | logit | 32.8 | 32.8 |
| fr | r | 29.9 | 31.3 |
| wikitext | j | 38.1 | 38.0 |
| wikitext | jpp | 55.8 | 57.2 |
| wikitext | logit | 32.8 | 32.8 |
| wikitext | r | 43.1 | 43.4 |

## Recall@10 at 1000 prompts with 95% bootstrap intervals over items (filtered)

| run | lens | recall10 | ci_low | ci_high |
|---|---|---|---|---|
| en-a | j | 37.9 | 33.5 | 42.4 |
| en-a | jpp | 57.1 | 52.9 | 61.0 |
| en-a | logit | 32.8 | 28.8 | 36.5 |
| en-a | r | 37.7 | 33.6 | 41.5 |
| en-b | j | 37.6 | 33.3 | 41.9 |
| en-b | jpp | 58.7 | 55.1 | 62.0 |
| en-b | logit | 32.8 | 28.8 | 36.6 |
| en-b | r | 37.7 | 33.6 | 41.6 |
| fr-a | j | 30.5 | 26.6 | 34.2 |
| fr-a | jpp | 48.7 | 44.4 | 53.0 |
| fr-a | logit | 32.8 | 28.9 | 36.5 |
| fr-a | r | 29.9 | 25.7 | 34.1 |
| fr-b | j | 30.5 | 26.2 | 34.5 |
| fr-b | jpp | 47.7 | 43.3 | 52.0 |
| fr-b | logit | 32.8 | 28.8 | 36.7 |
| fr-b | r | 31.3 | 27.4 | 35.1 |
| wikitext-a | j | 38.1 | 33.5 | 42.2 |
| wikitext-a | jpp | 55.8 | 51.4 | 60.0 |
| wikitext-a | logit | 32.8 | 28.8 | 36.5 |
| wikitext-a | r | 43.1 | 39.4 | 46.8 |
| wikitext-b | j | 38.0 | 33.8 | 42.0 |
| wikitext-b | jpp | 57.2 | 53.1 | 61.1 |
| wikitext-b | logit | 32.8 | 28.7 | 36.6 |
| wikitext-b | r | 43.4 | 39.4 | 46.8 |

## Recall@10 at 1000 prompts, paper protocol (J, R and Logit unfiltered; mean of pools)

| lens | en | fr | wikitext |
|---|---|---|---|
| j | 31.7 | 24.0 | 33.0 |
| jpp | 57.9 | 48.2 | 56.5 |
| logit | 31.0 | 31.0 | 31.0 |
| r | 31.2 | 26.0 | 37.0 |

## Distances at 1000 prompts, eval samples (each kind averaged over its pairs)

| family | lens | cosine | frobenius_rel | kl_sym |
|---|---|---|---|---|
| corpus | j | 0.895 | 0.457 | 1.982 |
| corpus | jpp | 0.682 | 0.954 | 4.775 |
| corpus | r | 0.939 | 0.337 | 1.128 |
| language | j | 0.834 | 0.570 | 1.907 |
| language | jpp | 0.707 | 0.970 | 3.538 |
| language | r | 0.890 | 0.452 | 0.968 |
| mixed | j | 0.767 | 0.697 | 3.709 |
| mixed | jpp | 0.665 | 0.980 | 5.525 |
| mixed | r | 0.853 | 0.538 | 2.026 |
| noise | j | 0.995 | 0.090 | 0.031 |
| noise | jpp | 0.795 | 0.753 | 3.119 |
| noise | r | 0.998 | 0.058 | 0.009 |

## Symmetric KL at 1000 prompts by sample set

| family | lens | en | eval | fr | wikitext |
|---|---|---|---|---|---|
| corpus | j | 1.33 | 1.98 | 1.40 | 1.49 |
| corpus | jpp | 3.59 | 4.77 | 3.81 | 4.21 |
| corpus | r | 1.01 | 1.13 | 1.17 | 1.33 |
| language | j | 2.29 | 1.91 | 1.96 | 2.00 |
| language | jpp | 3.54 | 3.54 | 3.38 | 3.99 |
| language | r | 2.77 | 0.97 | 1.97 | 2.41 |
| noise | j | 0.04 | 0.03 | 0.04 | 0.03 |
| noise | jpp | 2.39 | 3.12 | 2.56 | 2.86 |
| noise | r | 0.01 | 0.01 | 0.01 | 0.01 |

## Noise floor (A vs B, same dataset) by prompts: cosine

| kind | lens | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|
| noise-en | j | 0.952 | 0.977 | 0.992 | 0.995 |
| noise-en | jpp | 0.720 | 0.727 | 0.726 | 0.778 |
| noise-en | r | 0.973 | 0.988 | 0.996 | 0.998 |
| noise-fr | j | 0.985 | 0.992 | 0.996 | 0.994 |
| noise-fr | jpp | 0.735 | 0.690 | 0.706 | 0.757 |
| noise-fr | r | 0.996 | 0.998 | 0.999 | 0.998 |
| noise-wikitext | j | 0.963 | 0.983 | 0.993 | 0.996 |
| noise-wikitext | jpp | 0.785 | 0.893 | 0.884 | 0.849 |
| noise-wikitext | r | 0.975 | 0.989 | 0.996 | 0.998 |

## Layers (of 7) where an effect is above the noise floor on both cosine and KL

Each effect has two pairs (A and B), so the maximum count is 14 for language and corpus, 7 for mixed.

| family | lens | size | en | eval | fr | wikitext |
|---|---|---|---|---|---|---|
| corpus | j | 100 | 14 | 14 | 14 | 14 |
| corpus | j | 200 | 14 | 14 | 14 | 14 |
| corpus | j | 500 | 14 | 14 | 14 | 14 |
| corpus | j | 1000 | 14 | 14 | 14 | 14 |
| corpus | jpp | 100 | 8 | 7 | 8 | 8 |
| corpus | jpp | 200 | 8 | 8 | 4 | 7 |
| corpus | jpp | 500 | 8 | 8 | 9 | 8 |
| corpus | jpp | 1000 | 11 | 9 | 11 | 10 |
| corpus | r | 100 | 14 | 14 | 14 | 14 |
| corpus | r | 200 | 14 | 14 | 14 | 14 |
| corpus | r | 500 | 14 | 14 | 14 | 14 |
| corpus | r | 1000 | 14 | 14 | 14 | 14 |
| language | j | 100 | 14 | 14 | 14 | 14 |
| language | j | 200 | 14 | 14 | 14 | 14 |
| language | j | 500 | 14 | 14 | 14 | 14 |
| language | j | 1000 | 14 | 14 | 14 | 14 |
| language | jpp | 100 | 8 | 8 | 8 | 8 |
| language | jpp | 200 | 6 | 6 | 7 | 8 |
| language | jpp | 500 | 9 | 6 | 9 | 8 |
| language | jpp | 1000 | 10 | 9 | 10 | 8 |
| language | r | 100 | 14 | 14 | 14 | 14 |
| language | r | 200 | 14 | 14 | 14 | 14 |
| language | r | 500 | 14 | 14 | 14 | 14 |
| language | r | 1000 | 14 | 14 | 14 | 14 |
| mixed | j | 100 | 7 | 7 | 7 | 7 |
| mixed | j | 200 | 7 | 7 | 7 | 7 |
| mixed | j | 500 | 7 | 7 | 7 | 7 |
| mixed | j | 1000 | 7 | 7 | 7 | 7 |
| mixed | jpp | 100 | 3 | 1 | 4 | 3 |
| mixed | jpp | 200 | 4 | 3 | 4 | 4 |
| mixed | jpp | 500 | 6 | 4 | 6 | 6 |
| mixed | jpp | 1000 | 6 | 6 | 6 | 5 |
| mixed | r | 100 | 7 | 7 | 7 | 7 |
| mixed | r | 200 | 7 | 7 | 7 | 7 |
| mixed | r | 500 | 7 | 7 | 7 | 7 |
| mixed | r | 1000 | 7 | 7 | 7 | 7 |

## Convergence: cosine to the 1000-prompt lens of the same pool

| kind | lens | 100 | 200 | 500 |
|---|---|---|---|---|
| converge-en-a | j | 0.983 | 0.991 | 0.999 |
| converge-en-a | jpp | 0.879 | 0.896 | 0.973 |
| converge-en-a | r | 0.992 | 0.995 | 0.999 |
| converge-fr-a | j | 0.993 | 0.996 | 0.998 |
| converge-fr-a | jpp | 0.952 | 0.967 | 0.955 |
| converge-fr-a | r | 0.998 | 0.999 | 1.000 |
| converge-wikitext-a | j | 0.990 | 0.992 | 0.998 |
| converge-wikitext-a | jpp | 0.959 | 0.930 | 0.975 |
| converge-wikitext-a | r | 0.994 | 0.995 | 0.999 |

## Cross-method distances at 1000 prompts (eval samples)

| kind | j-jpp | j-r | r-jpp |
|---|---|---|---|
| method-en-a | 0.608 | 0.905 | 0.703 |
| method-en-b | 0.662 | 0.906 | 0.752 |
| method-fr-a | 0.579 | 0.884 | 0.679 |
| method-fr-b | 0.593 | 0.885 | 0.695 |
| method-wikitext-a | 0.626 | 0.894 | 0.740 |
| method-wikitext-b | 0.666 | 0.895 | 0.757 |

## KL from the model's own output to each lens (noise pairs' lenses, 1000 prompts)

| lens | en | eval | fr | wikitext |
|---|---|---|---|---|
| j | 8.45 | 7.80 | 9.05 | 8.51 |
| jpp | 7.81 | 8.36 | 9.39 | 7.50 |
| r | 5.63 | 5.99 | 6.66 | 5.65 |
