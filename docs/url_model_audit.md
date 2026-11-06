# URL model audit

Audit date: 2026-09-27. This report describes the URL pipeline and artifacts present before the new URL candidate is trained. No data in the final test partition was read for model selection during this audit.

## Data

The only training dataset currently present is UCI PhiUSIIL (`data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`), donated to UCI in 2024. It has 235,795 rows and 54 columns. UCI documents 134,850 legitimate and 100,945 phishing URLs; the source target is `1=legitimate, 0=phishing`, reversed by this project to `0=legitimate, 1=phishing`. The CSV has no per-row observation timestamp. The checked-in cleaned inspection reports 235,362 rows after removing 425 duplicate URL rows and 8 invalid or over-limit rows. Its cleaned class counts are 134,850 legitimate and 100,512 phishing. Fifty-eight supplied `Domain` values have both labels.

`data/hard_negative_urls.csv` is a separate, manually curated challenge set of 10 URLs, all labeled legitimate. It covers university payment pages, public login/support pages, help pages, and one reserved-domain lexical probe. It is not a representative population sample and currently has too few domains for stable category estimates.

## Model and features

The deployed real URL artifact is `backend/models/url_model.joblib` (about 4 MB), version 2.0.0, a calibrated character TF-IDF plus Logistic Regression pipeline. It consumes the raw URL and 35 URL-only numeric features. Character TF-IDF uses 2–5 character n-grams, `min_df=2`, at most 100,000 features, and sublinear term frequency. Logistic Regression uses `C=2`, `max_iter=1200`, balanced class weights, and a fixed random seed. The numeric feature set includes URL/host/path/query lengths, punctuation and digit counts/ratios, host labels, HTTPS, IP and punycode flags, ports, fragments, query count, percent encoding, suspicious-word counts by URL part, and character entropy. No submitted website is fetched or executed.

## Split and leakage audit

The recorded split is nested `StratifiedGroupKFold`: approximately 60% fit (141,867 rows), 20% validation (47,087), and 20% final test (46,408). It groups on PhiUSIIL's supplied `Domain` string, and the artifact reports zero exact-group overlap between partitions. This prevents leakage for repeated identical `Domain` values, but the code does not reduce hosts to registrable root domains; sibling subdomains can cross partitions. Exact URLs are deduplicated before splitting. There is no temporal holdout because the data has no row timestamps. The optional row-random benchmark is not recorded in the checked-in comparison; the primary reported results are grouped.

Model comparison and threshold selection use the validation partition. Calibration uses sigmoid/Platt scaling, fit on the same validation partition used to assess calibration and select thresholds. This does not touch the final test set, but the validation calibration figures are optimistic because they are in-sample for the calibrator. The final test is then used once for model metrics and calibration diagnostics.

## Baseline evaluation and operating thresholds

At a 0.5 model-probability threshold on grouped validation, the selected character model reported 99.72% accuracy, 99.99% precision, 99.36% recall, 99.67% F1, ROC-AUC 0.9987, PR-AUC 0.9990, and FPR 0.0074%. The selected operational malicious threshold was 0.9999964, producing final grouped-test accuracy 97.70%, precision 100%, recall 94.52%, F1 97.18%, FPR 0%, FNR 5.48%, ROC-AUC 0.9986, PR-AUC 0.9989, Brier score 0.00221, and confusion matrix TN 26,910 / FP 0 / FN 1,068 / TP 18,430. These results describe this dataset and split only.

The earlier curated hard-negative result was 0/10 safe at the model-warning threshold (100% any-warning FPR), while 1/10 received MALICIOUS after the structural guard (10% severe-warning FPR). The model therefore had a material false-positive problem despite the benchmark score. The existing severity guard requires two structural signals for a MALICIOUS URL verdict. A 0.70 minimum warning threshold has since been applied as an application policy; its performance must be re-evaluated in the candidate workflow rather than inferred from the older report.

## Current pipeline findings

- The trainer compares numeric Logistic Regression, Random Forest, HistGradientBoosting, and combined character TF-IDF plus numeric Logistic Regression. It does not yet compare a word-plus-character model or calibrated Linear SVM.
- The hard-negative report includes individual URLs but reports no per-category confusion matrices, and the ten samples are too small to support a general false-positive estimate.
- The URL inference path uses the same 35 numeric features and raw URL text for the selected model. Numeric-only model artifacts receive the numeric feature frame.
- Existing artifacts include the model, feature schema, dataset inspection, comparison, metrics, confusion matrix, and hard-negative results. Metadata does not include all library versions or latency/memory measurements.
- Model training and inference are scikit-learn based; the API loads artifacts lazily and keeps the Flask contract.

## Source notes

- [UCI PhiUSIIL dataset 967](https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset) identifies the source as CC BY 4.0 and documents its labels, row totals, and 2024 donation date.
- [PhishTank developer documentation](https://phishtank.org/developer_info.php) describes verified-online feeds and submission/verification timestamps. The current workflow has no registered feed key or downloaded snapshot; it is not silently downloaded. The preparation tool can import a user-provided, authorized CSV after the user acknowledges the current [PhishTank/Cisco terms](https://phishtank.org/terms.php).
- [OpenPhish terms](https://www.openphish.com/terms.html) restrict use and redistribution; its feed is not included. Researchers can request access through the [academic program](https://openphish.com/academic_use.html).
- [Tranco](https://tranco-list.eu/) combines ranking inputs with varying source licenses, including CC BY-SA and CC BY-NC inputs; no Tranco rows are incorporated into this dataset.

A temporal evaluation is only run if a supplied corpus has usable timestamps for every row and the temporal partitions can each retain both labels. The present UCI snapshot has no per-row observation timestamps.
