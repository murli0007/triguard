# URL model improvement report

Candidate: `url_model_v4.joblib` (combined_char_numeric_logistic_regression, version 4.0.0-candidate). The existing default model was preserved.

## Dataset and split

Rows: 233085; legitimate 133372; phishing 99713; registrable domains 174305.

Split: registrable-domain-disjoint StratifiedGroupKFold; 60/10/10/20 folds. Rows by partition: {'fit': 146688, 'calibration': 19863, 'validation': 20675, 'test': 45859}. Domains do not cross fit, calibration, validation, or test.
Separate hard-negative partitions: fit 1302 rows / 1284 domains; validation 21 / 21 domains; final challenge test 21 / 21 domains. Hard-negative roots were removed from the broad UCI corpus. Fit examples receive sample weight 10.

No timestamped rows were available, so this run has no temporal result.

## Model comparison on the same validation domains

| Model | F1 at 0.50 | PR-AUC | FPR | Brier | Mean ms | P95 ms | Compressed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| char_tfidf_logistic_regression | 0.9945 | 0.9984 | 0.0000% | 0.00337 | 0.622 | 0.791 | 1690623 |
| word_char_tfidf_logistic_regression | 0.9948 | 0.9982 | 0.0150% | 0.00346 | 0.745 | 0.796 | 2731620 |
| char_tfidf_linear_svm | 0.9958 | 0.9987 | 0.0000% | 0.00255 | 0.627 | 0.822 | 1697173 |
| numeric_random_forest | 0.9951 | 0.9980 | 0.0900% | 0.00326 | 13.977 | 14.661 | 11214954 |
| numeric_hist_gradient_boosting | 0.9949 | 0.9978 | 0.0600% | 0.00319 | 3.024 | 3.948 | 152947 |
| combined_char_numeric_logistic_regression | 0.9962 | 0.9985 | 0.0225% | 0.00255 | 1.118 | 1.223 | 1694424 |

## v2, v3, and improved-candidate comparison

All three artifacts are scored on the same UCI final rows and fresh external legitimate challenge rows. The v4 threshold is selected on validation only; v2/v3 use their previously saved thresholds. Because v2/v3 trained on the underlying UCI corpus, their UCI rows are retrospective and may overlap legacy training. The fresh external challenge roots do not overlap UCI or the prior hard-negative domains. No per-domain exception is applied.

| Version | Threshold | Accuracy | Phishing precision | Phishing recall | Phishing F1 | UCI FPR | UCI FNR | Hard-negative FPR | Mean inference ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 (url_char_tfidf_logistic_regression) | 0.70 | 0.9975 | 1.0000 | 0.9940 | 0.9970 | 0.000% | 0.603% | 100.00% | 1.106 | 1.174 |
| v3 (char_tfidf_logistic_regression) | 0.70 | 0.9977 | 1.0000 | 0.9944 | 0.9972 | 0.000% | 0.561% | 28.57% | 0.646 | 0.813 |
| v4 (combined_char_numeric_logistic_regression) | 0.80 | 0.9965 | 0.9999 | 0.9915 | 0.9957 | 0.004% | 0.852% | 14.29% | 1.118 | 1.223 |

### Final confusion matrices

| Version | UCI TN/FP/FN/TP | Hard-negative TN/FP/FN/TP |
|---|---|---|
| v2 | [[26953, 0], [114, 18792]] | [[0, 21], [0, 0]] |
| v3 | [[26953, 0], [106, 18800]] | [[15, 6], [0, 0]] |
| v4 | [[26952, 1], [161, 18745]] | [[18, 3], [0, 0]] |

## Selected candidate and untouched test

Selected on validation only: `combined_char_numeric_logistic_regression` at threshold 0.80. Rule: Highest phishing recall subject to UCI validation FPR <= 2% and hard-negative validation FPR <= 10%; ties prefer lower hard-negative FPR, then PR-AUC and F1. Both final test partitions are excluded from selection.

| Metric | Test result |
|---|---:|
| accuracy | 0.9965 |
| precision | 0.9999 |
| recall | 0.9915 |
| f1 | 0.9957 |
| roc_auc | 0.9989 |
| pr_auc | 0.9991 |
| false_positive_rate | 0.0037% |
| false_negative_rate | 0.8516% |
| brier_score | 0.0022 |
| Support | {'legitimate': 26953, 'phishing': 18906} |
| Confusion matrix TN/FP/FN/TP | [[26952, 1], [161, 18745]] |

### Existing model on the same final test

At the candidate threshold 0.80, existing model F1=0.9970, FPR=0.0000%, FNR=0.6030%, confusion=[[26953, 0], [114, 18792]]. This same-test comparison was computed after candidate selection and did not affect it.

## Separate hard-negative results

Fresh external official legitimate challenge URLs on unseen roots: 21. Candidate false-positive rate at its selected validation threshold: 14.29%.
Existing v2 model false-positive rate on the same examples at its saved threshold: 100.00%.
Hard-negative validation FPR at the selected threshold: 9.52%. Model and threshold selection prioritized hard-negative validation FPR under the broad-validation FPR cap; neither final test set informed selection.

| Hard-negative category | Samples | FPR | Confusion matrix TN/FP/FN/TP |
|---|---:|---:|---:|
| banking_home | 10 | 20.00% | [[8, 2], [0, 0]] |
| enterprise_identity | 1 | 0.00% | [[1, 0], [0, 0]] |
| enterprise_login | 1 | 100.00% | [[0, 1], [0, 0]] |
| government_account | 1 | 0.00% | [[1, 0], [0, 0]] |
| government_services | 4 | 0.00% | [[4, 0], [0, 0]] |
| university_portal | 4 | 0.00% | [[4, 0], [0, 0]] |

## Calibration

Selected method: sigmoid. Sigmoid validation Brier: 0.0025479381813740794; isotonic validation Brier: 0.0024651441400714647; calibration partition is separate from validation. Final-test Brier: 0.00224.

## Limitations and recommendations

The underlying UCI dataset is a static 2024 snapshot and has no per-URL timestamps. This is domain-generalization evaluation, not a measure of current live phishing coverage. The hard-negative challenge is mined from the UCI legitimate class and is not a population sample; independently verified recent benign URLs would provide a stronger challenge. The candidate is saved separately and is not promoted automatically. For production, add a dated, authorized phishing feed and a separately sourced dated benign URL stream, preserve label snapshots, retrain on older dates, and evaluate the newest date window without threshold tuning on it.
