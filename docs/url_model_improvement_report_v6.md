# URL model improvement report

Candidate: `url_model_v6.joblib` (combined_char_numeric_logistic_regression, version 6.0.0-candidate). The existing default model was preserved.

## Dataset and split

Rows: 234802; legitimate 134764; phishing 100038; registrable domains 175481.

Split: registrable-domain-disjoint StratifiedGroupKFold; 60/10/10/20 folds. Rows by partition: {'fit': 143450, 'calibration': 24433, 'validation': 21142, 'test': 45777}. Domains do not cross fit, calibration, validation, or test.
Separate hard-negative partitions: fit 96 rows / 95 domains; validation 23 / 23 domains; final challenge test 23 / 23 domains. Hard-negative roots were removed from the broad UCI corpus. Fit examples receive sample weight 10.

No timestamped rows were available, so this run has no temporal result.

## Model comparison on the same validation domains

| Model | F1 at 0.50 | PR-AUC | FPR | Brier | Mean ms | P95 ms | Compressed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| char_tfidf_logistic_regression | 0.9955 | 0.9979 | 0.0294% | 0.00315 | 0.613 | 0.676 | 1683813 |
| char_tfidf_no_generic_logistic_regression | 0.8158 | 0.9105 | 3.3703% | 0.08989 | 0.643 | 0.950 | 1683146 |
| word_char_tfidf_logistic_regression | 0.9947 | 0.9977 | 0.0147% | 0.00370 | 1.117 | 1.811 | 2670391 |
| word_char_no_generic_logistic_regression | 0.8269 | 0.9156 | 3.5906% | 0.08645 | 0.746 | 0.796 | 2668809 |
| char_tfidf_linear_svm | 0.9958 | 0.9983 | 0.0367% | 0.00268 | 0.614 | 0.689 | 1689718 |
| numeric_random_forest | 0.9842 | 0.9959 | 0.8224% | 0.00928 | 22.739 | 16.376 | 24570194 |
| numeric_hist_gradient_boosting | 0.9834 | 0.9958 | 0.4038% | 0.00938 | 10.214 | 13.553 | 187652 |
| combined_char_numeric_logistic_regression | 0.9948 | 0.9982 | 0.0881% | 0.00332 | 1.140 | 1.473 | 1687633 |

## v2, v3, and improved-candidate comparison

All three artifacts are scored on the same UCI final rows and fresh external legitimate challenge rows. The v4 threshold is selected on validation only; v2/v3 use their previously saved thresholds. Because v2/v3 trained on the underlying UCI corpus, their UCI rows are retrospective and may overlap legacy training. The fresh external challenge roots do not overlap UCI or the prior hard-negative domains. No per-domain exception is applied.

| Version | Threshold | Accuracy | Phishing precision | Phishing recall | Phishing F1 | UCI FPR | UCI FNR | Hard-negative FPR | Mean inference ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 (url_char_tfidf_logistic_regression) | 0.70 | 0.9925 | 0.9999 | 0.9820 | 0.9909 | 0.007% | 1.799% | 34.78% | 1.219 | 1.350 |
| v3 (char_tfidf_logistic_regression) | 0.70 | 0.9967 | 0.9999 | 0.9921 | 0.9960 | 0.004% | 0.794% | 34.78% | 0.804 | 1.026 |
| v6 (combined_char_numeric_logistic_regression) | 0.71 | 0.9954 | 0.9997 | 0.9890 | 0.9944 | 0.019% | 1.095% | 4.35% | 1.140 | 1.473 |

### Final confusion matrices

| Version | UCI TN/FP/FN/TP | Hard-negative TN/FP/FN/TP |
|---|---|---|
| v2 | [[26872, 2], [340, 18563]] | [[15, 8], [0, 0]] |
| v3 | [[26873, 1], [150, 18753]] | [[15, 8], [0, 0]] |
| v6 | [[26869, 5], [207, 18696]] | [[22, 1], [0, 0]] |

## Selected candidate and untouched test

Selected on validation only: `combined_char_numeric_logistic_regression` at threshold 0.71. Rule: Highest phishing recall subject to UCI validation FPR <= 2% and hard-negative validation FPR <= 10%; ties prefer lower hard-negative FPR, then PR-AUC and F1. Both final test partitions are excluded from selection.

| Metric | Test result |
|---|---:|
| accuracy | 0.9954 |
| precision | 0.9997 |
| recall | 0.9890 |
| f1 | 0.9944 |
| roc_auc | 0.9984 |
| pr_auc | 0.9987 |
| false_positive_rate | 0.0186% |
| false_negative_rate | 1.0951% |
| brier_score | 0.0034 |
| Support | {'legitimate': 26874, 'phishing': 18903} |
| Confusion matrix TN/FP/FN/TP | [[26869, 5], [207, 18696]] |

### Existing model on the same final test

At the candidate threshold 0.71, existing model F1=0.9909, FPR=0.0074%, FNR=1.7987%, confusion=[[26872, 2], [340, 18563]]. This same-test comparison was computed after candidate selection and did not affect it.

## Separate hard-negative results

Fresh external official legitimate challenge URLs on unseen roots: 23. Candidate false-positive rate at its selected validation threshold: 4.35%.
Existing v2 model false-positive rate on the same examples at its saved threshold: 34.78%.
Hard-negative validation FPR at the selected threshold: 8.70%. Model and threshold selection prioritized hard-negative validation FPR under the broad-validation FPR cap; neither final test set informed selection.

| Hard-negative category | Samples | FPR | Confusion matrix TN/FP/FN/TP |
|---|---:|---:|---:|
| account_management | 4 | 25.00% | [[3, 1], [0, 0]] |
| banking_home | 4 | 0.00% | [[4, 0], [0, 0]] |
| banking_login | 1 | 0.00% | [[1, 0], [0, 0]] |
| enterprise_login | 2 | 0.00% | [[2, 0], [0, 0]] |
| financial_services | 1 | 0.00% | [[1, 0], [0, 0]] |
| government_account | 2 | 0.00% | [[2, 0], [0, 0]] |
| government_authentication | 1 | 0.00% | [[1, 0], [0, 0]] |
| government_services | 2 | 0.00% | [[2, 0], [0, 0]] |
| university_portal | 6 | 0.00% | [[6, 0], [0, 0]] |

## Calibration

Selected method: sigmoid. Sigmoid validation Brier: 0.003322638021374623; isotonic validation Brier: 0.0032414747181014774; calibration partition is separate from validation. Final-test Brier: 0.00344.

## Limitations and recommendations

The underlying UCI dataset is a static 2024 snapshot and has no per-URL timestamps. This is domain-generalization evaluation, not a measure of current live phishing coverage. The hard-negative challenge is mined from the UCI legitimate class and is not a population sample; independently verified recent benign URLs would provide a stronger challenge. The candidate is saved separately and is not promoted automatically. For production, add a dated, authorized phishing feed and a separately sourced dated benign URL stream, preserve label snapshots, retrain on older dates, and evaluate the newest date window without threshold tuning on it.
