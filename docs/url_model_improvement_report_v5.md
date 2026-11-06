# URL model improvement report v5

v5 is a candidate only. The current default artifact remains v2. Candidate fitting and threshold selection used the v5 development hard-negative file and the registrable-domain-disjoint UCI fit/calibration/validation partitions. The challenge's registrable roots were used only in a separate data audit to prevent overlap; no challenge URL text, label, score, or performance result entered fit, calibration, or threshold selection. Challenge predictions were generated only after v5 was saved.

## v5 training and split

Selected model: `word_char_tfidf_logistic_regression`; threshold: **0.79**. Hard-negative development data: 96 fit, 23 validation, and 23 development-holdout rows. The challenge remains separate and has 21 registrable domains.

Threshold selection maximized phishing recall subject to the validation FPR caps in the trainer. The final UCI and challenge evaluations below did not affect model or threshold selection.

## v2 vs v3 vs v4 vs v5

UCI figures are measured on the same v5 registrable-domain-disjoint final rows. v2-v4 were scored retrospectively at their saved thresholds and may have seen some UCI data in their earlier training. The v5 challenge FPR uses the same 21 held-out legitimate URLs for all four artifacts and each artifact's saved threshold.

| Version | Threshold | UCI accuracy | Phishing precision | Phishing recall | Phishing F1 | UCI FPR | UCI FNR | Challenge FPs | Challenge FPR | Mean ms | P95 ms | Calibration / UCI Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v2 (url_char_tfidf_logistic_regression) | 0.70 | 99.7462% | 99.9878% | 99.3415% | 99.6636% | 0.0074% | 0.6585% | 21/21 | 100.00% | 1.131 | 1.221 | sigmoid / 0.00212 |
| v3 (char_tfidf_logistic_regression) | 0.70 | 99.7210% | 99.9939% | 99.2690% | 99.6301% | 0.0037% | 0.7310% | 6/21 | 28.57% | 0.639 | 0.729 | isotonic / 0.00218 |
| v4 (combined_char_numeric_logistic_regression) | 0.80 | 99.6799% | 99.9878% | 99.1663% | 99.5753% | 0.0074% | 0.8337% | 3/21 | 14.29% | 1.136 | 1.243 | sigmoid / 0.00210 |
| v5 (word_char_tfidf_logistic_regression) | 0.79 | 99.6456% | 99.9878% | 99.0756% | 99.5296% | 0.0074% | 0.9244% | 1/21 | 4.76% | 0.761 | 0.898 | sigmoid / 0.00330 |

### Confusion matrices

| Version | UCI TN/FP/FN/TP | Challenge TN/FP/FN/TP |
|---|---|---|
| v2 | [[27181, 2], [109, 16443]] | [[0, 21], [0, 0]] |
| v3 | [[27182, 1], [121, 16431]] | [[15, 6], [0, 0]] |
| v4 | [[27181, 2], [138, 16414]] | [[18, 3], [0, 0]] |
| v5 | [[27181, 2], [153, 16399]] | [[20, 1], [0, 0]] |

## Calibration and latency

v5 calibration: sigmoid; validation Brier score 0.002670; v5 UCI final Brier score 0.003302. Challenge Brier score: 0.086552 (descriptive for 21 legitimate-only examples). Latency is the trainer's 30 single-URL samples including URL feature extraction and prediction.

v5 phishing recall on the UCI final set is 99.08%; it changed by -0.09 percentage points versus v4 on these same rows. This is an unseen-domain evaluation on a static corpus, not a live or temporal performance claim.

## Held-out hard-negative false positives

- `https://us.etrade.com/` — p=0.9991; category: banking_home.

## Decision

Do not promote v5 automatically. Review its remaining held-out false positives and the phishing-recall tradeoff. The shipped default stays v2; v5 remains opt-in.

## Data and limits

v5 development rows are in `data/hard_negative_urls_v5.csv`; none share a registrable domain with the 21-domain challenge. UCI source data has no per-URL timestamps, so results do not establish performance on current live URLs. The challenge is small and its FPR is descriptive, not a population estimate.
