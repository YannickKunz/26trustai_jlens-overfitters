# Language transfer (recall@100 in %, 95% bootstrap intervals over items)

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
| logit | none | 47.6 [35.0, 55.5] | 20.0 [5.0, 35.0] | 40.3 [22.6, 55.5] | 35.0 [20.0, 50.0] |
| j | wikitext | 76.3 [65.8, 86.8] | 25.0 [10.0, 40.0] | 48.2 [30.5, 65.8] | 35.0 [20.0, 50.0] |
| j | en | 85.5 [75.0, 94.7] | 25.0 [10.0, 40.0] | 58.7 [40.8, 76.6] | 35.0 [20.0, 50.0] |
| j | fr | 76.3 [65.8, 86.8] | 43.2 [25.3, 61.1] | 41.6 [25.3, 56.6] | 45.3 [30.0, 58.2] |
| r | wikitext | 78.9 [68.4, 89.5] | 25.0 [10.0, 40.0] | 63.4 [48.2, 76.3] | 35.0 [20.0, 50.0] |
| r | en | 80.4 [68.7, 90.8] | 25.0 [10.0, 40.0] | 63.4 [48.2, 76.3] | 35.0 [20.0, 50.0] |
| r | fr | 71.3 [56.1, 84.2] | 37.9 [20.3, 53.7] | 40.0 [25.0, 50.0] | 42.6 [27.6, 55.3] |
| jpp | wikitext | 90.8 [81.6, 98.7] | 22.5 [7.5, 37.5] | 85.5 [76.3, 93.4] | 30.0 [17.5, 42.5] |
| jpp | en | 93.4 [86.8, 98.7] | 25.0 [10.0, 40.0] | 81.8 [70.4, 92.1] | 40.0 [25.0, 50.0] |
| jpp | fr | 89.5 [80.3, 97.4] | 44.6 [27.0, 62.1] | 64.9 [49.9, 78.9] | 61.1 [47.0, 74.1] |
| jpp_wen | wikitext | 90.8 [81.6, 98.7] | 25.0 [10.0, 40.0] | 78.9 [68.4, 88.2] | 35.0 [20.0, 50.0] |
| jpp_wen | en | 93.4 [85.5, 100.0] | 26.3 [10.0, 41.3] | 92.2 [83.4, 98.7] | 35.0 [20.0, 50.0] |
| jpp_wen | fr | 85.5 [75.0, 94.7] | 41.8 [24.2, 58.4] | 57.0 [43.2, 69.9] | 51.7 [39.1, 63.2] |
| jpp_wfr | wikitext | 82.9 [73.7, 92.1] | 26.3 [10.0, 41.3] | 62.4 [48.3, 75.5] | 22.6 [7.8, 37.8] |
| jpp_wfr | en | 82.9 [73.7, 92.1] | 23.8 [8.8, 38.8] | 58.3 [44.5, 70.0] | 36.3 [21.3, 50.0] |
| jpp_wfr | fr | 81.7 [71.3, 90.9] | 53.4 [38.3, 67.2] | 52.0 [36.7, 67.1] | 58.4 [41.8, 72.9] |
| jpp_matched | wikitext | 90.8 [81.6, 98.7] | 25.0 [10.0, 40.0] | 78.9 [68.4, 88.2] | 35.0 [20.0, 50.0] |
| jpp_matched | en | 93.4 [85.5, 100.0] | 26.3 [10.0, 41.3] | 92.2 [83.4, 98.7] | 35.0 [20.0, 50.0] |
| jpp_matched | fr | 81.7 [71.3, 90.9] | 53.4 [38.3, 67.2] | 52.0 [36.7, 67.1] | 58.4 [41.8, 72.9] |

## parallel: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | EN prompt, EN targets | EN prompt, FR targets | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|---|---|
| corpus: WikiText fit - EN fit | j | -9.2 [-18.4, -2.6] | 0.0 [0.0, 0.0] | -10.5 [-21.1, -2.6] | 0.0 [-15.0, 15.0] |
| corpus: WikiText fit - EN fit | jpp | -2.6 [-6.6, 0.0] | -2.5 [-7.5, 0.0] | 3.7 [-4.1, 12.6] | -10.0 [-22.5, -2.5] |
| corpus: WikiText fit - EN fit | jpp_matched | -2.6 [-9.2, 2.6] | -1.3 [-3.9, 0.0] | -13.3 [-22.5, -4.2] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | -2.6 [-9.2, 2.6] | -1.3 [-3.9, 0.0] | -13.3 [-22.5, -4.2] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 0.0 [-6.6, 6.6] | 2.5 [0.0, 7.5] | 4.1 [-4.9, 11.8] | -13.7 [-27.4, -2.4] |
| corpus: WikiText fit - EN fit | r | -1.4 [-9.2, 6.1] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 9.2 [-1.3, 19.7] | -18.2 [-33.2, -5.3] | 17.1 [5.3, 28.9] | -10.3 [-22.9, 0.0] |
| language: EN fit - FR fit | jpp | 3.9 [-2.6, 10.5] | -19.6 [-30.1, -9.2] | 17.0 [6.6, 27.5] | -21.1 [-32.6, -9.6] |
| language: EN fit - FR fit | jpp_matched | 11.7 [3.9, 20.9] | -27.1 [-40.1, -15.5] | 40.3 [27.1, 54.2] | -23.4 [-37.6, -11.8] |
| language: EN fit - FR fit | jpp_wen | 7.9 [1.3, 15.8] | -15.5 [-28.4, -5.3] | 35.3 [23.7, 45.8] | -16.7 [-29.3, -6.4] |
| language: EN fit - FR fit | jpp_wfr | 1.2 [-6.7, 10.4] | -29.6 [-43.2, -17.0] | 6.3 [-8.0, 21.8] | -22.1 [-35.8, -10.5] |
| language: EN fit - FR fit | r | 9.1 [-2.6, 21.1] | -12.9 [-25.8, -2.6] | 23.4 [10.5, 38.7] | -7.6 [-20.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 2.6 [0.0, 7.9] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 2.6 [-5.3, 10.5] | 0.0 [0.0, 0.0] | 10.0 [-5.3, 27.4] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | -2.6 [-7.9, 0.0] | 2.6 [0.0, 7.9] | -0.3 [-10.5, 12.4] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wen | -2.6 [-7.9, 0.0] | 2.6 [0.0, 7.9] | -0.3 [-10.5, 12.4] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | 7.9 [0.0, 15.8] | 7.6 [0.0, 20.3] | 20.8 [7.9, 36.1] | 2.6 [0.0, 7.9] |
| noise: EN fit, pool A - pool B | r | -2.4 [-15.0, 5.5] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | -5.3 [-13.2, 0.0] | 0.0 [0.0, 0.0] | -2.6 [-7.9, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 5.3 [0.0, 13.2] | 7.6 [-5.3, 20.5] | 7.6 [-2.9, 20.8] | -15.8 [-33.7, 1.8] |
| noise: FR fit, pool A - pool B | jpp_matched | 10.3 [-2.6, 25.5] | -10.5 [-30.5, 9.7] | 12.9 [2.6, 25.8] | -10.5 [-21.1, -2.6] |
| noise: FR fit, pool A - pool B | jpp_wen | -2.6 [-7.9, 0.0] | -2.6 [-10.5, 5.3] | -2.9 [-13.4, 9.7] | 2.4 [-10.8, 17.6] |
| noise: FR fit, pool A - pool B | jpp_wfr | 10.3 [-2.6, 25.5] | -10.5 [-30.5, 9.7] | 12.9 [2.6, 25.8] | -10.5 [-21.1, -2.6] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## parallel: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | EN prompt | FR prompt |
|---|---|---|
| j | 27.4 [9.2, 47.1] | 27.4 [9.2, 46.8] |
| jpp | 23.6 [11.8, 35.3] | 38.0 [22.5, 53.6] |
| jpp_matched | 38.8 [23.4, 55.5] | 63.7 [47.1, 80.4] |
| jpp_wen | 23.4 [11.6, 37.6] | 52.0 [36.4, 67.4] |
| jpp_wfr | 30.8 [15.1, 47.5] | 28.4 [12.0, 45.9] |
| r | 22.0 [6.1, 38.7] | 31.1 [15.2, 49.5] |

## fr-multilingual: recall by fitting corpus and condition (11 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 54.5 [38.6, 68.2] | 38.6 [31.8, 45.5] |
| j | wikitext | 68.2 [52.3, 84.1] | 59.1 [52.3, 65.9] |
| j | en | 80.7 [69.3, 92.0] | 56.8 [47.7, 65.9] |
| j | fr | 52.3 [36.4, 65.9] | 59.1 [52.3, 65.9] |
| r | wikitext | 89.8 [79.5, 97.7] | 54.5 [50.0, 61.4] |
| r | en | 75.0 [67.0, 84.1] | 50.0 [40.9, 59.1] |
| r | fr | 60.2 [50.0, 69.3] | 62.5 [54.5, 70.5] |
| jpp | wikitext | 88.6 [77.3, 97.7] | 68.2 [59.1, 77.3] |
| jpp | en | 94.3 [86.4, 100.0] | 73.9 [63.6, 84.1] |
| jpp | fr | 80.7 [71.6, 89.8] | 83.0 [75.0, 90.9] |
| jpp_wen | wikitext | 90.9 [79.5, 100.0] | 65.9 [54.5, 77.3] |
| jpp_wen | en | 92.0 [84.1, 98.9] | 71.6 [59.1, 83.0] |
| jpp_wen | fr | 69.3 [59.1, 78.4] | 62.5 [54.5, 71.6] |
| jpp_wfr | wikitext | 94.3 [88.6, 100.0] | 45.5 [35.2, 55.7] |
| jpp_wfr | en | 78.4 [70.5, 85.2] | 47.7 [44.3, 50.0] |
| jpp_wfr | fr | 63.6 [51.1, 75.0] | 89.8 [83.0, 96.6] |
| jpp_matched | wikitext | 90.9 [79.5, 100.0] | 65.9 [54.5, 77.3] |
| jpp_matched | en | 92.0 [84.1, 98.9] | 71.6 [59.1, 83.0] |
| jpp_matched | fr | 63.6 [51.1, 75.0] | 89.8 [83.0, 96.6] |

## fr-multilingual: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | -12.5 [-25.0, -2.3] | 2.3 [0.0, 6.8] |
| corpus: WikiText fit - EN fit | jpp | -5.7 [-11.4, -1.1] | -5.7 [-10.2, -1.1] |
| corpus: WikiText fit - EN fit | jpp_matched | -1.1 [-4.5, 2.3] | -5.7 [-13.6, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | -1.1 [-4.5, 2.3] | -5.7 [-13.6, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 15.9 [10.2, 21.6] | -2.3 [-11.4, 6.8] |
| corpus: WikiText fit - EN fit | r | 14.8 [8.0, 21.6] | 4.5 [0.0, 11.4] |
| language: EN fit - FR fit | j | 28.4 [18.2, 37.5] | -2.3 [-6.8, 0.0] |
| language: EN fit - FR fit | jpp | 13.6 [8.0, 19.3] | -9.1 [-14.8, -3.4] |
| language: EN fit - FR fit | jpp_matched | 28.4 [21.6, 36.4] | -18.2 [-33.0, -6.8] |
| language: EN fit - FR fit | jpp_wen | 22.7 [15.9, 31.8] | 9.1 [-6.8, 22.7] |
| language: EN fit - FR fit | jpp_wfr | 14.8 [5.7, 25.0] | -42.0 [-50.0, -34.1] |
| language: EN fit - FR fit | r | 14.8 [6.8, 25.0] | -12.5 [-26.1, 0.0] |
| noise: EN fit, pool A - pool B | j | 2.3 [0.0, 6.8] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 6.8 [0.0, 13.6] | -2.3 [-9.1, 4.5] |
| noise: EN fit, pool A - pool B | jpp_matched | -2.3 [-9.1, 4.5] | 2.3 [0.0, 6.8] |
| noise: EN fit, pool A - pool B | jpp_wen | -2.3 [-9.1, 4.5] | 2.3 [0.0, 6.8] |
| noise: EN fit, pool A - pool B | jpp_wfr | 11.4 [0.0, 22.7] | 0.0 [-6.8, 6.8] |
| noise: EN fit, pool A - pool B | r | 0.0 [-6.8, 6.8] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 2.3 [-4.5, 9.1] | 2.3 [-4.5, 9.1] |
| noise: FR fit, pool A - pool B | jpp_matched | 13.6 [4.5, 22.7] | -2.3 [-6.8, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | -20.5 [-29.5, -11.4] | -2.3 [-6.8, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 13.6 [4.5, 22.7] | -2.3 [-6.8, 0.0] |
| noise: FR fit, pool A - pool B | r | -2.3 [-6.8, 0.0] | -2.3 [-6.8, 0.0] |

## fr-multilingual: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 30.7 [20.5, 42.0] |
| jpp | 22.7 [13.6, 33.0] |
| jpp_matched | 46.6 [34.1, 61.4] |
| jpp_wen | 13.6 [-1.1, 29.6] |
| jpp_wfr | 56.8 [47.7, 68.2] |
| r | 27.3 [12.5, 44.3] |

## fr-multilingual, language name: recall by fitting corpus and condition (11 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| j | wikitext | 36.4 [9.1, 63.6] | 0.0 [0.0, 0.0] |
| j | en | 59.1 [31.8, 86.4] | 0.0 [0.0, 0.0] |
| j | fr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| r | wikitext | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] |
| r | en | 40.9 [13.6, 68.2] | 0.0 [0.0, 0.0] |
| r | fr | 0.0 [0.0, 0.0] | 4.5 [0.0, 13.6] |
| jpp | wikitext | 100.0 [100.0, 100.0] | 54.5 [31.8, 77.3] |
| jpp | en | 100.0 [100.0, 100.0] | 63.6 [36.4, 86.4] |
| jpp | fr | 77.3 [50.0, 100.0] | 72.7 [50.0, 95.5] |
| jpp_wen | wikitext | 100.0 [100.0, 100.0] | 63.6 [36.4, 90.9] |
| jpp_wen | en | 95.5 [86.4, 100.0] | 68.2 [40.9, 90.9] |
| jpp_wen | fr | 36.4 [22.7, 50.0] | 4.5 [0.0, 13.6] |
| jpp_wfr | wikitext | 100.0 [100.0, 100.0] | 13.6 [0.0, 27.3] |
| jpp_wfr | en | 54.5 [36.4, 72.7] | 0.0 [0.0, 0.0] |
| jpp_wfr | fr | 13.6 [0.0, 27.3] | 100.0 [100.0, 100.0] |
| jpp_matched | wikitext | 100.0 [100.0, 100.0] | 63.6 [36.4, 90.9] |
| jpp_matched | en | 95.5 [86.4, 100.0] | 68.2 [40.9, 90.9] |
| jpp_matched | fr | 13.6 [0.0, 27.3] | 100.0 [100.0, 100.0] |

## fr-multilingual, language name: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | -22.7 [-50.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | 0.0 [0.0, 0.0] | -9.1 [-22.7, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | 4.5 [0.0, 13.6] | -4.5 [-13.6, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | 4.5 [0.0, 13.6] | -4.5 [-13.6, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 45.5 [27.3, 63.6] | 13.6 [0.0, 27.3] |
| corpus: WikiText fit - EN fit | r | 59.1 [31.8, 86.4] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 59.1 [31.8, 86.4] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp | 22.7 [0.0, 50.0] | -9.1 [-22.7, 0.0] |
| language: EN fit - FR fit | jpp_matched | 81.8 [68.2, 95.5] | -31.8 [-59.1, -9.1] |
| language: EN fit - FR fit | jpp_wen | 59.1 [50.0, 72.7] | 63.6 [36.4, 86.4] |
| language: EN fit - FR fit | jpp_wfr | 40.9 [18.2, 68.2] | -100.0 [-100.0, -100.0] |
| language: EN fit - FR fit | r | 40.9 [13.6, 68.2] | -4.5 [-13.6, 0.0] |
| noise: EN fit, pool A - pool B | j | 9.1 [0.0, 27.3] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | -18.2 [-45.5, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | -9.1 [-27.3, 0.0] | 9.1 [0.0, 27.3] |
| noise: EN fit, pool A - pool B | jpp_wen | -9.1 [-27.3, 0.0] | 9.1 [0.0, 27.3] |
| noise: EN fit, pool A - pool B | jpp_wfr | 54.5 [27.3, 81.8] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | -9.1 [-27.3, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | -9.1 [-27.3, 0.0] | 0.0 [-27.3, 27.3] |
| noise: FR fit, pool A - pool B | jpp_matched | 27.3 [0.0, 54.5] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | -72.7 [-100.0, -45.5] | -9.1 [-27.3, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 27.3 [0.0, 54.5] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | -9.1 [-27.3, 0.0] |

## fr-multilingual, language name: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 59.1 [31.8, 86.4] |
| jpp | 31.8 [9.1, 54.5] |
| jpp_matched | 113.6 [86.4, 145.5] |
| jpp_wen | -4.5 [-36.4, 27.3] |
| jpp_wfr | 140.9 [118.2, 168.2] |
| r | 45.5 [13.6, 77.3] |

## fr-multilingual, category: recall by fitting corpus and condition (10 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 60.0 [30.0, 90.0] | 0.0 [0.0, 0.0] |
| j | wikitext | 70.0 [40.0, 100.0] | 50.0 [20.0, 80.0] |
| j | en | 80.0 [50.0, 100.0] | 40.0 [10.0, 70.0] |
| j | fr | 60.0 [30.0, 90.0] | 40.0 [10.0, 70.0] |
| r | wikitext | 80.0 [50.0, 100.0] | 40.0 [10.0, 70.0] |
| r | en | 80.0 [50.0, 100.0] | 20.0 [0.0, 50.0] |
| r | fr | 80.0 [50.0, 100.0] | 50.0 [20.0, 80.0] |
| jpp | wikitext | 80.0 [50.0, 100.0] | 25.0 [0.0, 50.0] |
| jpp | en | 85.0 [70.0, 100.0] | 35.0 [10.0, 65.0] |
| jpp | fr | 70.0 [45.0, 95.0] | 55.0 [25.0, 85.0] |
| jpp_wen | wikitext | 80.0 [50.0, 100.0] | 0.0 [0.0, 0.0] |
| jpp_wen | en | 85.0 [65.0, 100.0] | 25.0 [0.0, 50.0] |
| jpp_wen | fr | 70.0 [40.0, 100.0] | 50.0 [20.0, 80.0] |
| jpp_wfr | wikitext | 80.0 [50.0, 100.0] | 15.0 [5.0, 30.0] |
| jpp_wfr | en | 75.0 [50.0, 100.0] | 0.0 [0.0, 0.0] |
| jpp_wfr | fr | 75.0 [50.0, 100.0] | 55.0 [25.0, 85.0] |
| jpp_matched | wikitext | 80.0 [50.0, 100.0] | 0.0 [0.0, 0.0] |
| jpp_matched | en | 85.0 [65.0, 100.0] | 25.0 [0.0, 50.0] |
| jpp_matched | fr | 75.0 [50.0, 100.0] | 55.0 [25.0, 85.0] |

## fr-multilingual, category: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | -10.0 [-30.0, 0.0] | 10.0 [0.0, 30.0] |
| corpus: WikiText fit - EN fit | jpp | -5.0 [-20.0, 10.0] | -10.0 [-25.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | -5.0 [-15.0, 0.0] | -25.0 [-50.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | -5.0 [-15.0, 0.0] | -25.0 [-50.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 5.0 [0.0, 15.0] | 15.0 [5.0, 30.0] |
| corpus: WikiText fit - EN fit | r | 0.0 [0.0, 0.0] | 20.0 [0.0, 50.0] |
| language: EN fit - FR fit | j | 20.0 [0.0, 50.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp | 15.0 [0.0, 30.0] | -20.0 [-40.0, -5.0] |
| language: EN fit - FR fit | jpp_matched | 10.0 [0.0, 25.0] | -30.0 [-55.0, -10.0] |
| language: EN fit - FR fit | jpp_wen | 15.0 [0.0, 35.1] | -25.0 [-50.0, 0.0] |
| language: EN fit - FR fit | jpp_wfr | 0.0 [-15.0, 15.0] | -55.0 [-85.0, -25.0] |
| language: EN fit - FR fit | r | 0.0 [0.0, 0.0] | -30.0 [-60.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 30.0 [0.0, 60.0] | 10.0 [0.0, 30.0] |
| noise: EN fit, pool A - pool B | jpp_matched | 10.0 [0.0, 30.0] | 10.0 [0.0, 30.0] |
| noise: EN fit, pool A - pool B | jpp_wen | 10.0 [0.0, 30.0] | 10.0 [0.0, 30.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | -10.0 [-30.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 20.0 [0.0, 50.0] | 10.0 [0.0, 30.0] |
| noise: FR fit, pool A - pool B | jpp_matched | 10.0 [0.0, 30.0] | -10.0 [-30.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 10.0 [0.0, 30.0] | -10.0 [-30.0, 0.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## fr-multilingual, category: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 20.0 [0.0, 50.0] |
| jpp | 35.0 [15.0, 55.0] |
| jpp_matched | 40.0 [15.0, 75.0] |
| jpp_wen | 40.0 [15.0, 70.0] |
| jpp_wfr | 55.0 [20.0, 95.0] |
| r | 30.0 [0.0, 60.0] |

## fr-multilingual, operand: recall by fitting corpus and condition (11 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 72.7 [45.5, 100.0] | 54.5 [27.3, 81.8] |
| j | wikitext | 72.7 [45.5, 100.0] | 90.9 [72.7, 100.0] |
| j | en | 81.8 [54.5, 100.0] | 90.9 [72.7, 100.0] |
| j | fr | 63.6 [36.4, 90.9] | 100.0 [100.0, 100.0] |
| r | wikitext | 77.3 [54.5, 100.0] | 81.8 [54.5, 100.0] |
| r | en | 77.3 [54.5, 100.0] | 81.8 [54.5, 100.0] |
| r | fr | 68.2 [40.9, 90.9] | 100.0 [100.0, 100.0] |
| jpp | wikitext | 72.7 [45.5, 100.0] | 90.9 [77.3, 100.0] |
| jpp | en | 90.9 [72.7, 100.0] | 90.9 [72.7, 100.0] |
| jpp | fr | 72.7 [45.5, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wen | wikitext | 81.8 [54.5, 100.0] | 90.9 [72.7, 100.0] |
| jpp_wen | en | 86.4 [68.2, 100.0] | 86.4 [68.2, 100.0] |
| jpp_wen | fr | 72.7 [45.5, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wfr | wikitext | 95.5 [86.4, 100.0] | 59.1 [36.4, 81.8] |
| jpp_wfr | en | 86.4 [68.2, 100.0] | 90.9 [77.3, 100.0] |
| jpp_wfr | fr | 68.2 [40.9, 90.9] | 100.0 [100.0, 100.0] |
| jpp_matched | wikitext | 81.8 [54.5, 100.0] | 90.9 [72.7, 100.0] |
| jpp_matched | en | 86.4 [68.2, 100.0] | 86.4 [68.2, 100.0] |
| jpp_matched | fr | 68.2 [40.9, 90.9] | 100.0 [100.0, 100.0] |

## fr-multilingual, operand: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | -9.1 [-27.3, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | -18.2 [-45.5, 0.0] | 0.0 [-13.6, 13.6] |
| corpus: WikiText fit - EN fit | jpp_matched | -4.5 [-13.6, 0.0] | 4.5 [0.0, 13.6] |
| corpus: WikiText fit - EN fit | jpp_wen | -4.5 [-13.6, 0.0] | 4.5 [0.0, 13.6] |
| corpus: WikiText fit - EN fit | jpp_wfr | 9.1 [0.0, 27.3] | -31.8 [-54.5, -9.1] |
| corpus: WikiText fit - EN fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 18.2 [0.0, 45.5] | -9.1 [-27.3, 0.0] |
| language: EN fit - FR fit | jpp | 18.2 [0.0, 45.5] | -9.1 [-27.3, 0.0] |
| language: EN fit - FR fit | jpp_matched | 18.2 [0.0, 36.5] | -13.6 [-31.8, 0.0] |
| language: EN fit - FR fit | jpp_wen | 13.6 [0.0, 31.8] | -13.6 [-31.8, 0.0] |
| language: EN fit - FR fit | jpp_wfr | 18.2 [0.0, 36.5] | -9.1 [-22.7, 0.0] |
| language: EN fit - FR fit | r | 9.1 [0.0, 22.7] | -18.2 [-45.5, 0.0] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | -9.1 [-27.3, 0.0] | -9.1 [-27.3, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wen | -9.1 [-27.3, 0.0] | -9.1 [-27.3, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | -9.1 [-27.3, 0.0] | 0.0 [-27.3, 27.3] |
| noise: EN fit, pool A - pool B | r | 9.1 [0.0, 27.3] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_matched | 9.1 [0.0, 27.3] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 9.1 [0.0, 27.3] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | r | -9.1 [-27.3, 0.0] | 0.0 [0.0, 0.0] |

## fr-multilingual, operand: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 27.3 [0.0, 54.5] |
| jpp | 27.3 [0.0, 54.5] |
| jpp_matched | 31.8 [9.1, 54.5] |
| jpp_wen | 27.3 [4.5, 50.0] |
| jpp_wfr | 27.3 [9.1, 45.5] |
| r | 27.3 [0.0, 59.1] |

## fr-multilingual, answer: recall by fitting corpus and condition (10 items)

| lens | fit | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| logit | none | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| j | wikitext | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| j | en | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| j | fr | 90.0 [70.0, 100.0] | 100.0 [100.0, 100.0] |
| r | wikitext | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| r | en | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| r | fr | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp | wikitext | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp | en | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp | fr | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wen | wikitext | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wen | en | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wen | fr | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wfr | wikitext | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wfr | en | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_wfr | fr | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_matched | wikitext | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_matched | en | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| jpp_matched | fr | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

## fr-multilingual, answer: differences in points (an interval excluding 0 is a detectable effect)

| contrast | lens | FR prompt, EN targets | FR prompt, FR targets |
|---|---|---|---|
| corpus: WikiText fit - EN fit | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | jpp_wfr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| corpus: WikiText fit - EN fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | j | 10.0 [0.0, 30.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | jpp_wfr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| language: EN fit - FR fit | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | jpp_wfr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: EN fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | j | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_matched | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wen | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | jpp_wfr | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| noise: FR fit, pool A - pool B | r | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

## fr-multilingual, answer: target-language specificity

(EN fit - FR fit on English targets) - (EN fit - FR fit on French targets), in points. Positive: each lens reads the words of its own fitting language better.

| lens | FR prompt |
|---|---|
| j | 10.0 [0.0, 30.0] |
| jpp | 0.0 [0.0, 0.0] |
| jpp_matched | 0.0 [0.0, 0.0] |
| jpp_wen | 0.0 [0.0, 0.0] |
| jpp_wfr | 0.0 [0.0, 0.0] |
| r | 0.0 [0.0, 0.0] |

## Transfer gap G = (in-language lens - other-language lens) / in-language lens

Domain = prompts and targets in that language, parallel items. The H1 rows are G(lens) - G(J): positive means the lens transfers worse than J.

| quantity | lens | estimate | share_above_0 |
|---|---|---|---|
| G, EN domain | j | 0.11 [-0.02, 0.22] | 0.95 |
| G, FR domain | j | 0.23 [0.00, 0.48] | 0.955 |
| G, mean | j | 0.17 [0.03, 0.31] | 0.994 |
| G, EN domain | r | 0.11 [-0.03, 0.26] | 0.92 |
| G, FR domain | r | 0.18 [0.00, 0.44] | 0.866 |
| G, mean | r | 0.15 [0.02, 0.30] | 0.988 |
| G, EN domain | jpp | 0.04 [-0.03, 0.12] | 0.844 |
| G, FR domain | jpp | 0.34 [0.18, 0.51] | 1.0 |
| G, mean | jpp | 0.19 [0.10, 0.28] | 1.0 |
| G, EN domain | jpp_wen | 0.08 [0.01, 0.17] | 0.989 |
| G, FR domain | jpp_wen | 0.32 [0.13, 0.56] | 1.0 |
| G, mean | jpp_wen | 0.20 [0.09, 0.33] | 1.0 |
| G, EN domain | jpp_wfr | 0.01 [-0.09, 0.13] | 0.546 |
| G, FR domain | jpp_wfr | 0.38 [0.20, 0.58] | 1.0 |
| G, mean | jpp_wfr | 0.20 [0.10, 0.31] | 1.0 |
| G, EN domain | jpp_matched | 0.13 [0.04, 0.22] | 0.999 |
| G, FR domain | jpp_matched | 0.40 [0.22, 0.61] | 1.0 |
| G, mean | jpp_matched | 0.26 [0.16, 0.38] | 1.0 |
| G mean - G mean of J (H1) | r | -0.02 [-0.12, 0.07] | 0.343 |
| G mean - G mean of J (H1) | jpp | 0.03 [-0.08, 0.13] | 0.701 |
| G mean - G mean of J (H1) | jpp_wen | 0.04 [-0.04, 0.14] | 0.805 |
| G mean - G mean of J (H1) | jpp_wfr | 0.03 [-0.06, 0.13] | 0.729 |
| G mean - G mean of J (H1) | jpp_matched | 0.10 [0.01, 0.21] | 0.982 |
| old G: FR prompt, EN targets (fr-multilingual) | j | -0.54 [-0.97, -0.30] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | r | -0.25 [-0.45, -0.10] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp | -0.17 [-0.26, -0.09] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp_wen | -0.33 [-0.53, -0.21] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp_wfr | -0.23 [-0.47, -0.08] | 0.0 |
| old G: FR prompt, EN targets (fr-multilingual) | jpp_matched | -0.45 [-0.70, -0.28] | 0.0 |

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
