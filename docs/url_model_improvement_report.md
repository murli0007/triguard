# URL model improvement report

Candidate: `url_model_v3.joblib` (char_tfidf_logistic_regression, version 3.0.0-candidate). The existing default model was preserved.

## Dataset and split

Rows: 234461; legitimate 134747; phishing 99714; registrable domains 175477.

Split: registrable-domain-disjoint StratifiedGroupKFold; 60/10/10/20 folds. Rows by partition: {'fit': 133206, 'calibration': 21038, 'validation': 31090, 'test': 49127}. Domains do not cross fit, calibration, validation, or test.
Separate hard-negative partitions: fit 26 rows / 24 domains; validation 8 / 8 domains; final challenge test 29 / 18 domains. Hard-negative roots were removed from the broad UCI corpus. Fit examples receive sample weight 100.

No timestamped rows were available, so this run has no temporal result.

## Model comparison on the same validation domains

| Model | F1 at 0.50 | PR-AUC | FPR | Brier | Mean ms | P95 ms | Compressed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| char_tfidf_logistic_regression | 0.9985 | 0.9996 | 0.0149% | 0.00154 | 0.614 | 0.664 | 1678855 |
| word_char_tfidf_logistic_regression | 0.9980 | 0.9996 | 0.0972% | 0.00202 | 0.763 | 0.840 | 2374555 |
| char_tfidf_linear_svm | 0.9988 | 0.9996 | 0.0224% | 0.00121 | 0.862 | 0.927 | 1680575 |
| numeric_random_forest | 0.9916 | 0.9991 | 0.0673% | 0.00757 | 16.182 | 16.985 | 9021450 |
| numeric_hist_gradient_boosting | 0.9906 | 0.9992 | 0.1121% | 0.00892 | 2.468 | 3.026 | 120532 |
| combined_char_numeric_logistic_regression | 0.9977 | 0.9996 | 0.0598% | 0.00209 | 1.403 | 1.474 | 1683175 |

## Selected candidate and untouched test

Selected on validation only: `char_tfidf_logistic_regression` at threshold 0.70. Rule: No model met both validation FPR constraints; selected by validation PR-AUC then F1. Final test and hard-negative test domains are excluded from selection.

| Metric | Test result |
|---|---:|
| accuracy | 0.9966 |
| precision | 1.0000 |
| recall | 0.9926 |
| f1 | 0.9962 |
| roc_auc | 0.9988 |
| pr_auc | 0.9988 |
| false_positive_rate | 0.0037% |
| false_negative_rate | 0.7444% |
| brier_score | 0.0025 |
| Support | {'legitimate': 27096, 'phishing': 22031} |
| Confusion matrix TN/FP/FN/TP | [[27095, 1], [164, 21867]] |

### Existing model on the same final test

At the candidate threshold 0.70, existing model F1=0.9973, FPR=0.0074%, FNR=0.5311%, confusion=[[27094, 2], [117, 21914]]. This same-test comparison was computed after candidate selection and did not affect it.

## Separate hard-negative results

Held-out legitimate challenge URLs (curated plus UCI-mined): 29. Candidate false-positive rate at its selected validation threshold: 62.07%.
Existing model false-positive rate on the same examples at 0.70: 89.66%.
Hard-negative validation FPR at the selected threshold: 50.00%. No model met both validation FPR limits in this run, so the comparison fell back to validation PR-AUC/F1; the selected candidate therefore remains opt-in.

| Hard-negative category | Samples | FPR | Confusion matrix TN/FP/FN/TP |
|---|---:|---:|---:|
| official_account | 3 | 100.00% | [[0, 3], [0, 0]] |
| official_content | 1 | 100.00% | [[0, 1], [0, 0]] |
| official_help | 4 | 100.00% | [[0, 4], [0, 0]] |
| official_home | 4 | 25.00% | [[3, 1], [0, 0]] |
| official_login | 10 | 70.00% | [[3, 7], [0, 0]] |
| official_payment | 4 | 50.00% | [[2, 2], [0, 0]] |
| uci_legitimate_suspicious_terms | 3 | 0.00% | [[3, 0], [0, 0]] |

## Calibration

Selected method: isotonic. Sigmoid validation Brier: 0.0015878098032928258; isotonic validation Brier: 0.0015421108741786682; calibration partition is separate from validation. Final-test Brier: 0.00251.

## Limitations and recommendations

The underlying UCI dataset is a static 2024 snapshot and has no per-URL timestamps. This is domain-generalization evaluation, not a measure of current live phishing coverage. The hard-negative challenge is small and partly mined from UCI, so it cannot establish population false-positive rates. A candidate is written separately and is not promoted into the default REAL endpoint unless explicitly selected with `URL_MODEL_VERSION=v3` after review. For production, add a dated, authorized phishing feed and a separately sourced dated benign URL stream, preserve label snapshots, retrain on older dates, and evaluate the newest date window without threshold tuning on it.
