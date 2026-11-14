# Language transfer (recall@10 in %, 95% bootstrap intervals over items)

Items kept at each filter:
- scored items: 94
- held-out: 65
- answered correctly in every prompt language used: 48
- with a concept that is a different, single-token word in both languages: 40
- parallel: 19 association, 10 multihop
- fr-multilingual: 11 multilingual
- fr-multilingual, language name: 11 multilingual
- fr-multilingual, category: 10 multilingual
- fr-multilingual, operand: 11 multilingual
- fr-multilingual, answer: 10 multilingual

`jpp`: the existing J++ lens. `jpp_wen` / `jpp_wfr`: expert weights fitted on English / French labels. `jpp_matched`: weights in the language of the lens's own corpus.

## parallel: recall by fitting corpus and condition (29 items)

| lens | fit | EN prompt, EN targets | EN prompt, FR targets | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|---|---|
| logit | none | 45.0 [35.0, 50.0] | 5.0 [0.0, 15.0] | 35.0 [20.0, 45.0] | 0.0 [0.0, 0.0] |
| j | wikitext | 47.9 [32.6, 60.8] | 0.0 [0.0, 0.0] | 37.6 [22.6, 50.5] | 0.0 [0.0, 0.0] |
| j | en | 54.5 [38.2, 68.7] | 0.0 [0.0, 0.0] | 41.6 [25.3, 56.8] | 0.0 [0.0, 0.0] |
| j | fr | 40.3 [25.0, 55.3] | 25.3 [10.0, 42.9] | 30.0 [15.0, 45.0] | 20.0 [5.0, 35.0] |
| r | wikitext | 50.3 [37.6, 60.5] | 0.0 [0.0, 0.0] | 37.6 [22.6, 50.5] | 0.0 [0.0, 0.0] |
| r | en | 45.1 [32.6, 55.3] | 0.0 [0.0, 0.0] | 35.0 [20.0, 45.0] | 0.0 [0.0, 0.0] |
| r | fr | 40.1 [27.5, 52.6] | 15.0 [0.0, 30.0] | 30.0 [15.0, 45.0] | 15.0 [0.0, 30.0] |
| jpp | wikitext | 68.8 [53.8, 81.7] | 0.0 [0.0, 0.0] | 50.8 [33.2, 67.4] | 0.0 [0.0, 0.0] |
| jpp | en | 77.9 [66.2, 88.4] | 0.0 [0.0, 0.0] | 57.2 [40.7, 72.8] | 0.0 [0.0, 0.0] |
| jpp | fr | 64.6 [54.2, 75.1] | 12.6 [5.0, 21.4] | 39.1 [22.9, 54.2] | 16.4 [7.6, 25.4] |
| jpp_wen | wikitext | 63.4 [50.3, 76.3] | 0.0 [0.0, 0.0] | 48.2 [31.6, 63.7] | 0.0 [0.0, 0.0] |
| jpp_wen | en | 79.2 [65.0, 91.1] | 0.0 [0.0, 0.0] | 53.6 [35.7, 71.3] | 0.0 [0.0, 0.0] |
| jpp_wen | fr | 53.0 [40.4, 64.6] | 18.8 [5.0, 33.8] | 40.1 [26.4, 51.6] | 18.8 [5.0, 33.8] |
| jpp_wfr | wikitext | 43.0 [27.8, 58.2] | 0.0 [0.0, 0.0] | 37.8 [22.6, 51.6] | 0.0 [0.0, 0.0] |
| jpp_wfr | en | 49.1 [37.6, 59.2] | 0.0 [0.0, 0.0] | 40.3 [25.3, 54.2] | 0.0 [0.0, 0.0] |
| jpp_wfr | fr | 48.0 [32.8, 62.2] | 20.1 [6.3, 35.1] | 31.3 [17.5, 43.9] | 23.8 [10.0, 38.9] |
| jpp_matched | wikitext | 63.4 [50.3, 76.3] | 0.0 [0.0, 0.0] | 48.2 [31.6, 63.7] | 0.0 [0.0, 0.0] |
| jpp_matched | en | 79.2 [65.0, 91.1] | 0.0 [0.0, 0.0] | 53.6 [35.7, 71.3] | 0.0 [0.0, 0.0] |
| jpp_matched | fr | 48.0 [32.8, 62.2] | 20.1 [6.3, 35.1] | 31.3 [17.5, 43.9] | 23.8 [10.0, 38.9] |

## parallel: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | EN prompt, EN targets | EN prompt, FR targets | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|---|---|
| corpus: WikiText fit - EN fit | j | -6.6 [-14.5, 0.0] | 0.0 [0.0, 0.0] | -3.9 [-10.5, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | -9.1 [-20.8, 2.4] | 0.0 [0.0, 0.0] | -6.4 [-20.5, 6.2] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | -15.8 [-27.6, -3.9] | 0.0 [0.0, 0.0] | -5.4 [-17.1, 5.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | -15.8 [-27.6, -3.9] | 0.0 [0.0, 0.0] | -5.4 [-17.1, 5.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | -6.1 [-22.4, 9.1] | 0.0 [0.0, 0.0] | -2.5 [-10.1, 3.9] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | r | 5.1 [0.0, 12.9] | 0.0 [0.0, 0.0] | 2.6 [0.0, 7.9] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 14.2 [2.6, 28.2] | -25.3 [-42.9, -10.0] | 11.6 [1.3, 24.7] | -20.0 [-35.0, -5.0] |
| language: EN fit - FR fit | jpp | 13.3 [1.7, 25.0] | -12.6 [-21.4, -5.0] | 18.2 [5.3, 32.4] | -16.4 [-25.4, -7.6] |
| language: EN fit - FR fit | jpp_matched | 31.2 [17.1, 46.3] | -20.1 [-35.1, -6.3] | 22.2 [10.5, 34.1] | -23.8 [-38.9, -10.0] |
| language: EN fit - FR fit | jpp_wen | 26.2 [11.0, 41.7] | -18.8 [-33.8, -5.0] | 13.4 [1.7, 25.1] | -18.8 [-33.8, -5.0] |
| language: EN fit - FR fit | jpp_wfr | 1.1 [-13.0, 16.1] | -20.1 [-35.1, -6.3] | 8.9 [2.5, 17.8] | -23.8 [-38.9, -10.0] |
| language: EN fit - FR fit | r | 5.0 [-5.3, 15.3] | -15.0 [-30.0, 0.0] | 5.0 [0.0, 15.0] | -15.0 [-30.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 2.6 [0.0, 7.9] | 0.0 [0.0, 0.0] | 2.6 [0.0, 7.9] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | -7.4 [-22.6, 5.5] | 0.0 [0.0, 0.0] | -2.4 [-15.5, 10.5] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | -5.3 [-13.2, 0.0] | 0.0 [0.0, 0.0] | -5.0 [-15.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wen | -5.3 [-13.2, 0.0] | 0.0 [0.0, 0.0] | -5.0 [-15.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | 12.9 [0.0, 28.2] | 0.0 [0.0, 0.0] | 10.5 [2.6, 21.1] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | -5.0 [-15.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | -7.6 [-20.8, 2.9] | 20.0 [4.5, 37.6] | 2.4 [-7.9, 15.0] | 17.1 [-0.5, 34.7] |
| noise: FR fit, pool A - pool B | jpp_matched | 0.3 [-12.4, 10.5] | -10.3 [-23.2, 0.0] | 2.6 [-12.4, 17.6] | -7.6 [-20.3, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | -5.5 [-18.4, 7.4] | 2.4 [-7.9, 15.0] | -0.3 [-10.5, 12.4] | 2.4 [-7.9, 15.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 0.3 [-12.4, 10.5] | -10.3 [-23.2, 0.0] | 2.6 [-12.4, 17.6] | -7.6 [-20.3, 0.0] |
| noise: FR fit, pool A - pool B | r | -5.0 [-15.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## parallel: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | EN prompt | FR prompt |
|---|---|---|
| j | 39.5 [16.3, 66.3] | 31.6 [15.0, 48.2] |
| jpp | 25.9 [11.7, 40.0] | 34.6 [20.4, 49.9] |
| jpp_matched | 51.3 [32.4, 70.0] | 46.1 [27.0, 66.7] |
| jpp_wen | 45.0 [24.7, 64.1] | 32.2 [12.9, 52.5] |
| jpp_wfr | 21.2 [1.2, 40.3] | 32.8 [15.0, 51.6] |
| r | 20.0 [4.9, 35.1] | 20.0 [5.0, 35.0] |

## fr-multilingual: recall by fitting corpus and condition (11 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 27.3 [18.2, 36.4] | 22.7 [15.9, 29.5] |
| j | wikitext | 54.5 [38.6, 68.2] | 27.3 [25.0, 31.8] |
| j | en | 54.5 [38.6, 68.2] | 27.3 [25.0, 31.8] |
| j | fr | 26.1 [14.8, 37.5] | 43.2 [34.1, 52.3] |
| r | wikitext | 52.3 [34.1, 68.2] | 22.7 [18.2, 25.0] |
| r | en | 56.8 [40.9, 70.5] | 27.3 [25.0, 31.8] |
| r | fr | 36.4 [18.2, 52.3] | 42.0 [31.8, 52.3] |
| jpp | wikitext | 77.3 [62.5, 89.8] | 25.0 [18.2, 31.8] |
| jpp | en | 78.4 [63.6, 90.9] | 26.1 [19.3, 31.8] |
| jpp | fr | 58.0 [40.9, 72.7] | 30.7 [25.0, 36.4] |
| jpp_wen | wikitext | 84.1 [70.5, 95.5] | 23.9 [18.2, 28.4] |
| jpp_wen | en | 79.5 [64.8, 93.2] | 26.1 [19.3, 31.8] |
| jpp_wen | fr | 30.7 [19.3, 42.0] | 36.4 [29.5, 43.2] |
| jpp_wfr | wikitext | 59.1 [42.0, 72.7] | 9.1 [3.4, 15.9] |
| jpp_wfr | en | 56.8 [44.3, 67.0] | 21.6 [17.0, 25.0] |
| jpp_wfr | fr | 30.7 [17.0, 44.3] | 67.0 [54.5, 79.5] |
| jpp_matched | wikitext | 84.1 [70.5, 95.5] | 23.9 [18.2, 28.4] |
| jpp_matched | en | 79.5 [64.8, 93.2] | 26.1 [19.3, 31.8] |
| jpp_matched | fr | 30.7 [17.0, 44.3] | 67.0 [54.5, 79.5] |

## fr-multilingual: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | -1.1 [-6.8, 3.4] | -1.1 [-4.5, 2.3] |
| corpus: WikiText fit - EN fit | jpp_matched | 4.5 [-1.1, 10.2] | -2.3 [-5.7, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | 4.5 [-1.1, 10.2] | -2.3 [-5.7, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 2.3 [-5.7, 9.1] | -12.5 [-18.2, -5.7] |
| corpus: WikiText fit - EN fit | r | -4.5 [-13.6, 0.0] | -4.5 [-13.6, 0.0] |
| language: EN fit - FR fit | j | 28.4 [18.2, 38.6] | -15.9 [-25.0, -6.8] |
| language: EN fit - FR fit | jpp | 20.5 [9.1, 34.1] | -4.5 [-12.5, 3.4] |
| language: EN fit - FR fit | jpp_matched | 48.9 [35.2, 63.6] | -40.9 [-55.7, -28.4] |
| language: EN fit - FR fit | jpp_wen | 48.9 [35.2, 63.6] | -10.2 [-20.5, -1.1] |
| language: EN fit - FR fit | jpp_wfr | 26.1 [17.0, 38.6] | -45.5 [-61.4, -30.7] |
| language: EN fit - FR fit | r | 20.5 [9.1, 31.8] | -14.8 [-25.0, -4.5] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | -6.8 [-13.6, 0.0] | -2.3 [-9.1, 4.5] |
| noise: EN fit, pool A - pool B | jpp_matched | 0.0 [-6.8, 6.8] | 6.8 [0.0, 13.6] |
| noise: EN fit, pool A - pool B | jpp_wen | 0.0 [-6.8, 6.8] | 6.8 [0.0, 13.6] |
| noise: EN fit, pool A - pool B | jpp_wfr | 18.2 [2.3, 34.1] | 2.3 [0.0, 6.8] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 2.3 [0.0, 6.8] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 2.3 [-9.1, 15.9] | 25.0 [11.4, 38.6] |
| noise: FR fit, pool A - pool B | jpp_matched | 15.9 [4.5, 27.3] | -6.8 [-20.5, 9.1] |
| noise: FR fit, pool A - pool B | jpp_wen | 6.8 [0.0, 13.6] | 0.0 [-6.8, 6.8] |
| noise: FR fit, pool A - pool B | jpp_wfr | 15.9 [4.5, 27.3] | -6.8 [-20.5, 9.1] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | -2.3 [-6.8, 0.0] |

## fr-multilingual: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 44.3 [29.5, 58.0] |
| jpp | 25.0 [9.1, 43.2] |
| jpp_matched | 89.8 [67.0, 118.2] |
| jpp_wen | 59.1 [42.0, 81.8] |
| jpp_wfr | 71.6 [50.0, 98.9] |
| r | 35.2 [22.7, 46.6] |

## fr-multilingual, language name: recall by fitting corpus and condition (11 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| j | wikitext | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| j | en | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| j | fr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| r | wikitext | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| r | en | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| r | fr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| jpp | wikitext | 86.4 [63.6, 100.0] | 9.1 [0.0, 22.7] |
| jpp | en | 81.8 [59.1, 100.0] | 9.1 [0.0, 22.7] |
| jpp | fr | 36.4 [13.6, 59.1] | 4.5 [0.0, 13.6] |
| jpp_wen | wikitext | 95.5 [86.4, 100.0] | 0.0 [0.0, 0.0] |
| jpp_wen | en | 86.4 [63.6, 100.0] | 9.1 [0.0, 22.7] |
| jpp_wen | fr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| jpp_wfr | wikitext | 27.3 [4.5, 50.0] | 0.0 [0.0, 0.0] |
| jpp_wfr | en | 22.7 [9.1, 36.4] | 0.0 [0.0, 0.0] |
| jpp_wfr | fr | 0.0 [0.0, 0.0] | 63.6 [45.5, 81.8] |
| jpp_matched | wikitext | 95.5 [86.4, 100.0] | 0.0 [0.0, 0.0] |
| jpp_matched | en | 86.4 [63.6, 100.0] | 9.1 [0.0, 22.7] |
| jpp_matched | fr | 0.0 [0.0, 0.0] | 63.6 [45.5, 81.8] |

## fr-multilingual, language name: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | 4.5 [0.0, 13.6] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | 9.1 [0.0, 22.7] | -9.1 [-22.7, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | 9.1 [0.0, 22.7] | -9.1 [-22.7, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 4.5 [-22.7, 31.8] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp | 45.5 [27.3, 63.6] | 4.5 [0.0, 13.6] |
| language: EN fit - FR fit | jpp_matched | 86.4 [63.6, 100.0] | -54.5 [-68.2, -40.9] |
| language: EN fit - FR fit | jpp_wen | 86.4 [63.6, 100.0] | 9.1 [0.0, 22.7] |
| language: EN fit - FR fit | jpp_wfr | 22.7 [9.1, 36.4] | -63.6 [-81.8, -45.5] |
| language: EN fit - FR fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | -18.2 [-45.5, 0.0] | -18.2 [-45.5, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | -9.1 [-27.3, 0.0] | 18.2 [0.0, 45.5] |
| noise: EN fit, pool A - pool B | jpp_wen | -9.1 [-27.3, 0.0] | 18.2 [0.0, 45.5] |
| noise: EN fit, pool A - pool B | jpp_wfr | 45.5 [18.2, 72.7] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | -18.2 [-54.5, 18.2] | 9.1 [0.0, 27.3] |
| noise: FR fit, pool A - pool B | jpp_matched | 0.0 [0.0, 0.0] | -36.4 [-72.7, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 0.0 [0.0, 0.0] | -36.4 [-72.7, 0.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## fr-multilingual, language name: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 0.0 [0.0, 0.0] |
| jpp | 40.9 [13.6, 63.6] |
| jpp_matched | 140.9 [104.5, 168.2] |
| jpp_wen | 77.3 [54.5, 95.5] |
| jpp_wfr | 86.4 [63.6, 104.5] |
| r | 0.0 [0.0, 0.0] |

## fr-multilingual, category: recall by fitting corpus and condition (10 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| j | wikitext | 60.0 [30.0, 90.0] | 0.0 [0.0, 0.0] |
| j | en | 60.0 [30.0, 90.0] | 0.0 [0.0, 0.0] |
| j | fr | 30.0 [0.0, 60.0] | 10.0 [0.0, 30.0] |
| r | wikitext | 70.0 [40.0, 100.0] | 0.0 [0.0, 0.0] |
| r | en | 70.0 [40.0, 100.0] | 0.0 [0.0, 0.0] |
| r | fr | 40.0 [10.0, 70.0] | 30.0 [0.0, 60.0] |
| jpp | wikitext | 65.0 [35.0, 90.0] | 0.0 [0.0, 0.0] |
| jpp | en | 65.0 [35.0, 90.0] | 0.0 [0.0, 0.0] |
| jpp | fr | 55.0 [25.0, 85.0] | 0.0 [0.0, 0.0] |
| jpp_wen | wikitext | 70.0 [40.0, 100.0] | 0.0 [0.0, 0.0] |
| jpp_wen | en | 65.0 [35.0, 90.0] | 0.0 [0.0, 0.0] |
| jpp_wen | fr | 35.0 [10.0, 65.0] | 10.0 [0.0, 30.0] |
| jpp_wfr | wikitext | 70.0 [40.0, 100.0] | 0.0 [0.0, 0.0] |
| jpp_wfr | en | 60.0 [30.0, 85.0] | 0.0 [0.0, 0.0] |
| jpp_wfr | fr | 35.0 [10.0, 60.0] | 30.0 [10.0, 50.0] |
| jpp_matched | wikitext | 70.0 [40.0, 100.0] | 0.0 [0.0, 0.0] |
| jpp_matched | en | 65.0 [35.0, 90.0] | 0.0 [0.0, 0.0] |
| jpp_matched | fr | 35.0 [10.0, 60.0] | 30.0 [10.0, 50.0] |

## fr-multilingual, category: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | 5.0 [-15.0, 30.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | 5.0 [-15.0, 30.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 10.0 [0.0, 25.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 30.0 [0.0, 60.0] | -10.0 [-30.0, 0.0] |
| language: EN fit - FR fit | jpp | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp_matched | 30.0 [10.0, 50.0] | -30.0 [-50.0, -10.0] |
| language: EN fit - FR fit | jpp_wen | 30.0 [5.0, 55.0] | -10.0 [-30.0, 0.0] |
| language: EN fit - FR fit | jpp_wfr | 25.0 [0.0, 50.0] | -30.0 [-50.0, -10.0] |
| language: EN fit - FR fit | r | 30.0 [0.0, 60.0] | -30.0 [-60.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | -10.0 [-30.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wen | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | 20.0 [0.0, 50.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_matched | 30.0 [0.0, 60.0] | 20.0 [-20.0, 60.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 30.0 [0.0, 60.0] | 20.0 [-20.0, 60.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## fr-multilingual, category: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 40.0 [10.0, 70.0] |
| jpp | 10.0 [0.0, 30.0] |
| jpp_matched | 60.0 [30.0, 90.0] |
| jpp_wen | 40.0 [15.0, 65.0] |
| jpp_wfr | 55.0 [20.0, 95.0] |
| r | 60.0 [30.0, 90.0] |

## fr-multilingual, operand: recall by fitting corpus and condition (11 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 18.2 [0.0, 45.5] | 9.1 [0.0, 27.3] |
| j | wikitext | 72.7 [45.5, 100.0] | 9.1 [0.0, 27.3] |
| j | en | 72.7 [45.5, 100.0] | 9.1 [0.0, 27.3] |
| j | fr | 4.5 [0.0, 13.6] | 63.6 [36.4, 90.9] |
| r | wikitext | 63.6 [36.4, 90.9] | 0.0 [0.0, 0.0] |
| r | en | 72.7 [45.5, 100.0] | 9.1 [0.0, 27.3] |
| r | fr | 36.4 [9.1, 63.6] | 40.9 [13.6, 68.2] |
| jpp | wikitext | 68.2 [40.9, 90.9] | 4.5 [0.0, 13.6] |
| jpp | en | 72.7 [45.5, 100.0] | 4.5 [0.0, 13.6] |
| jpp | fr | 63.6 [36.4, 90.9] | 36.4 [18.2, 54.5] |
| jpp_wen | wikitext | 72.7 [45.5, 100.0] | 4.5 [0.0, 13.6] |
| jpp_wen | en | 72.7 [45.5, 100.0] | 4.5 [0.0, 13.6] |
| jpp_wen | fr | 9.1 [0.0, 22.7] | 36.4 [13.6, 63.6] |
| jpp_wfr | wikitext | 54.5 [27.3, 81.8] | 0.0 [0.0, 0.0] |
| jpp_wfr | en | 59.1 [36.4, 81.8] | 0.0 [0.0, 0.0] |
| jpp_wfr | fr | 18.2 [4.5, 31.8] | 68.2 [40.9, 90.9] |
| jpp_matched | wikitext | 72.7 [45.5, 100.0] | 4.5 [0.0, 13.6] |
| jpp_matched | en | 72.7 [45.5, 100.0] | 4.5 [0.0, 13.6] |
| jpp_matched | fr | 18.2 [4.5, 31.8] | 68.2 [40.9, 90.9] |

## fr-multilingual, operand: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | -4.5 [-13.6, 0.0] | 0.0 [-13.6, 13.6] |
| corpus: WikiText fit - EN fit | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [-13.6, 13.6] |
| corpus: WikiText fit - EN fit | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [-13.6, 13.6] |
| corpus: WikiText fit - EN fit | jpp_wfr | -4.5 [-36.4, 22.7] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | r | -9.1 [-27.3, 0.0] | -9.1 [-27.3, 0.0] |
| language: EN fit - FR fit | j | 68.2 [40.9, 90.9] | -54.5 [-81.8, -27.3] |
| language: EN fit - FR fit | jpp | 9.1 [0.0, 27.3] | -31.8 [-50.0, -13.6] |
| language: EN fit - FR fit | jpp_matched | 54.5 [31.8, 77.3] | -63.6 [-86.4, -36.4] |
| language: EN fit - FR fit | jpp_wen | 63.6 [36.4, 86.4] | -31.8 [-59.1, -9.1] |
| language: EN fit - FR fit | jpp_wfr | 40.9 [22.7, 59.1] | -68.2 [-90.9, -40.9] |
| language: EN fit - FR fit | r | 36.4 [9.1, 63.6] | -31.8 [-59.1, -9.1] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 9.1 [0.0, 27.3] |
| noise: EN fit, pool A - pool B | jpp_matched | 0.0 [0.0, 0.0] | 9.1 [0.0, 27.3] |
| noise: EN fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 9.1 [0.0, 27.3] |
| noise: EN fit, pool A - pool B | jpp_wfr | -9.1 [-36.4, 18.2] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 9.1 [0.0, 27.3] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 54.5 [27.3, 81.8] |
| noise: FR fit, pool A - pool B | jpp_matched | 36.4 [9.1, 63.6] | -9.1 [-27.3, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 18.2 [0.0, 45.5] | 0.0 [-27.3, 27.3] |
| noise: FR fit, pool A - pool B | jpp_wfr | 36.4 [9.1, 63.6] | -9.1 [-27.3, 0.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | -9.1 [-27.3, 0.0] |

## fr-multilingual, operand: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 122.7 [72.7, 168.2] |
| jpp | 40.9 [18.2, 68.2] |
| jpp_matched | 118.2 [72.7, 163.6] |
| jpp_wen | 95.5 [50.0, 140.9] |
| jpp_wfr | 109.1 [68.2, 150.0] |
| r | 68.2 [27.3, 109.1] |

## fr-multilingual, answer: recall by fitting corpus and condition (10 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 90.0 [70.0, 100.0] | 90.0 [70.0, 100.0] |
| j | wikitext | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| j | en | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| j | fr | 80.0 [50.0, 100.0] | 100.0 [100.0, 100.0] |
| r | wikitext | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| r | en | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| r | fr | 80.0 [50.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp | wikitext | 90.0 [70.0, 100.0] | 95.0 [85.0, 100.0] |
| jpp | en | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp | fr | 85.0 [65.0, 100.0] | 85.0 [70.0, 100.0] |
| jpp_wen | wikitext | 95.0 [85.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wen | en | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wen | fr | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wfr | wikitext | 90.0 [70.0, 100.0] | 40.0 [15.0, 65.0] |
| jpp_wfr | en | 85.0 [65.0, 100.0] | 95.0 [85.0, 100.0] |
| jpp_wfr | fr | 80.0 [50.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_matched | wikitext | 95.0 [85.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_matched | en | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_matched | fr | 80.0 [50.0, 100.0] | 100.0 [100.0, 100.0] |

## fr-multilingual, answer: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | 0.0 [0.0, 0.0] | -5.0 [-15.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | 5.0 [0.0, 15.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | 5.0 [0.0, 15.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 5.0 [0.0, 15.0] | -55.0 [-80.0, -30.0] |
| corpus: WikiText fit - EN fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp | 5.0 [0.0, 15.0] | 15.0 [0.0, 30.0] |
| language: EN fit - FR fit | jpp_matched | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp_wfr | 5.0 [0.0, 15.0] | -5.0 [-15.0, 0.0] |
| language: EN fit - FR fit | r | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | 10.0 [0.0, 30.0] | 10.0 [0.0, 30.0] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 10.0 [0.0, 30.0] | 30.0 [0.0, 60.0] |
| noise: FR fit, pool A - pool B | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## fr-multilingual, answer: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 10.0 [0.0, 30.0] |
| jpp | -10.0 [-30.0, 10.0] |
| jpp_matched | 10.0 [0.0, 30.0] |
| jpp_wen | 0.0 [0.0, 0.0] |
| jpp_wfr | 10.0 [0.0, 25.0] |
| r | 10.0 [0.0, 30.0] |

## Transfer gap G = (in-language lens - other-language lens) / in-language lens

Domain = prompts and targets in that language, parallel items. The H1 rows are G(lens) - G(J): positive means the lens transfers worse than J.

| quantity | lens | estimate | share_above_0 |
|---|---|---|---|
| G, EN domain | j | 0.26 [0.07, 0.49] | 0.996 |
| G, FR domain | j | 1.00 [1.00, 1.00] | 1.0 |
| G, mean | j | 0.63 [0.53, 0.75] | 1.0 |
| G, EN domain | r | 0.11 [-0.11, 0.34] | 0.789 |
| G, FR domain | r | 1.00 [1.00, 1.00] | 1.0 |
| G, mean | r | 0.56 [0.45, 0.66] | 1.0 |
| G, EN domain | jpp | 0.17 [0.02, 0.30] | 0.993 |
| G, FR domain | jpp | 1.00 [1.00, 1.00] | 1.0 |
| G, mean | jpp | 0.59 [0.51, 0.65] | 1.0 |
| G, EN domain | jpp_wen | 0.33 [0.16, 0.49] | 1.0 |
| G, FR domain | jpp_wen | 1.00 [1.00, 1.00] | 1.0 |
| G, mean | jpp_wen | 0.67 [0.58, 0.75] | 1.0 |
| G, EN domain | jpp_wfr | 0.02 [-0.29, 0.30] | 0.529 |
| G, FR domain | jpp_wfr | 1.00 [1.00, 1.00] | 1.0 |
| G, mean | jpp_wfr | 0.51 [0.36, 0.65] | 1.0 |
| G, EN domain | jpp_matched | 0.39 [0.23, 0.56] | 1.0 |
| G, FR domain | jpp_matched | 1.00 [1.00, 1.00] | 1.0 |
| G, mean | jpp_matched | 0.70 [0.62, 0.78] | 1.0 |
| G mean - G mean of J (H1) | r | -0.08 [-0.16, 0.01] | 0.05 |
| G mean - G mean of J (H1) | jpp | -0.05 [-0.19, 0.07] | 0.259 |
| G mean - G mean of J (H1) | jpp_wen | 0.03 [-0.10, 0.15] | 0.716 |
| G mean - G mean of J (H1) | jpp_wfr | -0.12 [-0.27, 0.02] | 0.048 |
| G mean - G mean of J (H1) | jpp_matched | 0.07 [-0.03, 0.17] | 0.905 |
| old G: FR prompt, EN targets (fr-multilingual) | j | -1.09 [-2.07, -0.64] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | r | -0.56 [-1.40, -0.21] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp | -0.35 [-0.75, -0.14] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp_wen | -1.59 [-2.95, -0.97] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp_wfr | -0.85 [-2.00, -0.43] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp_matched | -1.59 [-3.35, -0.90] | 0.0 |

## Log-rank transfer gap (parallel items, bits)

Per domain d (prompts and targets in d): mean log2 best rank of the targets under the other-language lens minus under the d-language lens. +1 means the targets are ranked twice as low. Unlike recall@k it does not saturate. The H1 rows are gap(lens) - gap(J): positive means the lens transfers worse than J.

| quantity | lens | estimate | share_above_0 |
|---|---|---|---|
| log-rank gap, EN domain | j | 1.13 [0.60, 1.68] | 1.0 |
| log-rank gap, FR domain | j | 1.91 [1.42, 2.38] | 1.0 |
| log-rank gap, mean | j | 1.52 [1.08, 1.94] | 1.0 |
| log-rank gap, EN domain | r | 0.86 [0.46, 1.28] | 1.0 |
| log-rank gap, FR domain | r | 1.79 [1.37, 2.20] | 1.0 |
| log-rank gap, mean | r | 1.33 [0.98, 1.66] | 1.0 |
| log-rank gap, EN domain | jpp | 0.65 [0.23, 1.07] | 0.999 |
| log-rank gap, FR domain | jpp | 2.61 [2.14, 3.07] | 1.0 |
| log-rank gap, mean | jpp | 1.63 [1.39, 1.87] | 1.0 |
| log-rank gap, EN domain | jpp_wen | 1.42 [0.88, 2.03] | 1.0 |
| log-rank gap, FR domain | jpp_wen | 2.23 [1.67, 2.74] | 1.0 |
| log-rank gap, mean | jpp_wen | 1.82 [1.46, 2.23] | 1.0 |
| log-rank gap, EN domain | jpp_wfr | 0.18 [-0.30, 0.70] | 0.75 |
| log-rank gap, FR domain | jpp_wfr | 3.24 [2.56, 3.89] | 1.0 |
| log-rank gap, mean | jpp_wfr | 1.71 [1.30, 2.11] | 1.0 |
| log-rank gap, EN domain | jpp_matched | 1.96 [1.40, 2.58] | 1.0 |
| log-rank gap, FR domain | jpp_matched | 2.82 [2.17, 3.44] | 1.0 |
| log-rank gap, mean | jpp_matched | 2.39 [2.01, 2.80] | 1.0 |
| log-rank gap - that of J (H1) | r | -0.19 [-0.48, 0.10] | 0.095 |
| log-rank gap - that of J (H1) | jpp | 0.11 [-0.29, 0.53] | 0.7 |
| log-rank gap - that of J (H1) | jpp_wen | 0.30 [-0.13, 0.79] | 0.907 |
| log-rank gap - that of J (H1) | jpp_wfr | 0.19 [-0.21, 0.61] | 0.809 |
| log-rank gap - that of J (H1) | jpp_matched | 0.87 [0.45, 1.36] | 1.0 |
