# Email dataset and leakage audit for Model v2

**Audit date:** 2026-09-27  
**Scope:** read-only audit of the MeAJOR CSV, current derived CSV, email pipeline, and saved artifacts. No model was trained and no artifact or dataset was changed. The only created file is this report.

## Executive summary

The current email model’s reported 22.29% false-positive rate is measured at a fixed probability threshold of 0.50 on a TREC-05 source holdout. The model is a lexical word TF-IDF classifier, so it cannot distinguish legitimate corporate, account, financial, and notification language from similar phishing wording except through correlations learned from the training collections. TREC-05 is held out by source, but not by sender domain or subject/template. Calibration is also drawn randomly from the same two training sources, and the trainer does not select a classification threshold on an independent validation partition.

The saved metrics cannot be reproduced exactly from the CSV currently in the workspace: its cleaned corpus has 104,342 rows, while the saved artifact says 104,328; the TREC-05 counts differ by 11. The current model can be applied to the current TREC-05 rows for descriptive error analysis, but that is not an exact reproduction of the historical metrics. The persisted model also warns that it was created with scikit-learn 1.7.2 and is being loaded with 1.6.1 in this environment.

## 1. Dataset audit

### Sources and row counts

The downloaded MeAJOR file, `data/raw/email_curated/meajor_cleaned_preprocessed.csv`, is a 20-column CSV with 108,685 rows. Its raw label counts are 60,650 benign (`0`), 48,034 phishing (`1`), and one missing label. Among labeled rows, the class balance is 55.8% benign and 44.2% phishing.

The preparation script selects subject, body, source, and label, drops rows missing body or label, normalizes missing subjects to empty strings, and removes repeated exact subject/body pairs. The current workspace counts are:

| Stage/source | Rows | Benign | Phishing | Phishing share |
|---|---:|---:|---:|---:|
| Raw MeAJOR, TREC-05 | 49,583 | 29,929 | 19,654 | 39.6% |
| Raw MeAJOR, TREC-06 | 15,005 | 11,208 | 3,797 | 25.3% |
| Raw MeAJOR, TREC-07 | 44,096 | 19,513 | 24,583 | 55.8% |
| Prepared CSV, TREC-05 | 46,852 | 27,556 | 19,296 | 41.2% |
| Prepared CSV, TREC-06 | 14,705 | 11,190 | 3,515 | 23.9% |
| Prepared CSV, TREC-07 | 43,375 | 19,341 | 24,034 | 55.4% |
| Trainer-cleaned corpus, TREC-05 | 46,544 | 27,352 | 19,192 | 41.2% |
| Trainer-cleaned corpus, TREC-06 | 14,677 | 11,180 | 3,497 | 23.8% |
| Trainer-cleaned corpus, TREC-07 | 43,121 | 19,320 | 23,801 | 55.2% |
| **Trainer-cleaned total** | **104,342** | **57,852** | **46,490** | **44.6%** |

The current raw source class rates differ substantially: TREC-06 is mostly benign, TREC-07 is majority phishing, and TREC-05 is between them. This is a source/domain shift even before considering changes in vocabulary or mail style.

### Duplicates and preprocessing removals

- There are **13 exact duplicate full rows** in the raw file.
- There are **3,751 excess rows** sharing an exact subject/body pair in the raw file. The downloader removes repeated subject/body pairs before writing `data/raw/phishing_emails.csv`.
- Applying `clean_email_text` to that prepared file yields **590 excess duplicate cleaned-text rows**, which the trainer removes globally before splitting.
- The exact subject/body groups and the cleaned-text duplicate groups audited here have **zero conflicting-label groups**. This does not rule out individual label errors elsewhere.
- Two raw rows lack a required body or label and are dropped by preparation. Missing subjects are filled with empty strings. The prepared file has 104,932 rows; trainer cleaning leaves 104,342. Thus **4,343 rows are removed** from the 108,685-row raw file in the current run: two rows missing required fields, 3,751 subject/body duplicate rows, and 590 further cleaned-text duplicate rows. The 13 full-row duplicates are included in the subject/body duplicate count, not additive.

The 590 cleaned-text removals occur before splitting, so identical cleaned strings cannot appear on both sides of the trainer’s split. One exact cleaned-text duplicate group spans TREC-05 and TREC-06 before removal. Its retained source depends on row order; this can alter source counts even though it does not create a cross-split duplicate after deduplication.

## 2. TREC split and leakage audit

### Exact current split construction

`backend/train_email_model.py` calls `GroupShuffleSplit` with `test_size=0.20`, seed 42, and `source` as the only group. It retries up to 30 generated splits until both labels exist in each partition. The saved metadata confirms the selected final test source is TREC-05, with TREC-06 and TREC-07 in the non-test pool. The fitted model is trained on 49,125 rows. The saved 8,670-row “validation” partition is a stratified random 15% sample of the remaining source pool (seed 43); it is used to fit sigmoid calibration, not to choose a decision threshold. It is not domain- or template-disjoint from fit rows.

The saved metadata reports TREC-05 test support of 46,533 messages (27,352 benign and 19,181 phishing), fit 49,125, and calibration 8,670. The current trainer-cleaned CSV instead yields TREC-05 46,544 (27,352 benign and 19,192 phishing), fit-pool sources totaling 57,798, and total 104,342. Do not silently regenerate or substitute the saved final evaluation based on this changed row set; first pin the exact data snapshot and its checksum.

### Sender and receiver overlap

Counts below are aggregated. No email addresses or domain names are included in this report.

| Identifier | Distinct TREC-05 values | Distinct TREC-06/07 values | Shared distinct values | TREC-05 rows using a shared value |
|---|---:|---:|---:|---:|
| Anonymized sender identifier | 23,033 | 32,542 | 14 | 111 |
| Sender domain | 9,620 | 19,893 | 1,049 | 9,676 of 49,228 rows with a sender domain (19.7%) |
| Anonymized receiver identifier | 11,889 | 5,694 | 3 | 26 |
| Receiver domain | 3,419 | 3,006 | 59 | 649 of 48,005 rows with a receiver domain (1.4%) |

The TREC-05/TREC-06/07 sender-domain overlap is material. On the current trainer-cleaned data, **12,847 training-pool messages** have a sender domain that appears in TREC-05. Removing those rows would leave 44,951 training-pool messages (22,801 benign, 22,150 phishing), before allocating fit, calibration, and validation. Shared domains can be common mail providers, so overlap alone does not prove a same-sender leak; it does mean the reported test is not an unseen-domain evaluation.

The raw corpus contains 28,464 distinct sender domains; median messages per domain is 1, mean 3.8, and the largest domain has 21,684 messages. Of these domains, 21,348 occur once. At the domain level, 24,352 are observed only with phishing labels, 3,666 only with benign labels, and 446 with both. These label associations are strong potential shortcuts and may reflect dataset collection/source composition rather than generalizable domain risk. Receiver domains are similarly concentrated: 6,366 distinct values, median 1 and mean 16.7 messages per domain.

**Assessment:** grouping by sender domain would provide a more realistic *unseen-sender-domain* stress test. A strict comparison that keeps all TREC-05 messages fixed must exclude training-pool rows whose sender domains occur in TREC-05; otherwise domain overlap remains. Missing domains should be kept in an explicit “unknown” category and never grouped together as if they were the same domain.

## 3. Similarity and template audit

- Exact cleaned-text duplicates are removed globally before split, so none remain shared across fit/calibration/test in the current trainer flow.
- A normalized subject-only comparison finds **322 distinct subject strings** shared between TREC-05 and TREC-06/07, covering **4,473 TREC-05 rows**. Subject equality alone can be generic (for example, reply subjects) and is not proof of a repeated body or campaign.
- A lightweight exploratory near-duplicate screen used bottom-k signatures of word five-grams with locality-sensitive bands, comparing TREC-05 to TREC-06/07. With a fixed `PYTHONHASHSEED=0`, it identified **135 TREC-05 messages** whose candidate signature overlap was at least 0.75 (8,123 test messages had at least one LSH candidate). This is a candidate count, not a verified semantic similarity count: short/common messages and boilerplate can collide, and no message contents are reproduced here.
- The dataset has no campaign or thread identifier. Therefore campaign/template overlap cannot be exactly measured from a dedicated field. The subject and near-duplicate findings show that source disjointness alone does not establish campaign disjointness.

These results make similarity leakage plausible for some messages, but do not show that it explains most false positives or that the reported recall is inflated. A v2 split should group verified near-duplicate/template clusters before partitioning and report the resulting counts.

## 4. Feature and train/inference audit

### Text preprocessing

`backend/email_features.py` combines subject, newline, and body into one text string. It HTML-unescapes, feeds content through Python’s `HTMLParser`, keeps visible text nodes, and joins those nodes with spaces. It does not parse structured MIME parts or retain HTML attributes as text. It replaces detected `http(s)://...` and `www....` strings with the token `URL`, replaces email-address-shaped strings with `EMAIL`, removes most punctuation while keeping selected characters, collapses whitespace, and lowercases.

Consequences:

- Subject and body are concatenated and not separately weighted. Subject-only emphasis is not configured.
- The model sees visible URL presence as a placeholder but not the URL host, destination, or link-to-anchor mismatch.
- The cleaning URL pattern does not recognize bare domains without `www.` or a scheme.
- HTML attributes such as `href`, sender headers, MIME structure, attachments, and display-name/header mismatch are not features of the email text model.
- Email addresses are replaced in the text model. The standalone API accepts no sender field.

### Classifier and calibration

The trainer uses a scikit-learn pipeline with `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True)` and default word tokenization, L2 normalization, and IDF settings. The classifier is `LogisticRegression(max_iter=1800, C=1.0, class_weight="balanced", random_state=42)`. Class balancing affects the fitted decision boundary and does not itself minimize FPR.

The base classifier is fitted on fit rows. `CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")` then learns a sigmoid probability map from the separate calibration rows. The calibration partition is random by message and shares TREC sources/domains/templates with fit. No calibration data comes from TREC-05.

The training report computes metrics at **0.50**. There is no validation threshold sweep or selected threshold in the trainer. The production email route does not implement a `p >= 0.50` rule: it converts probability into generic risk bands (safe through 29, suspicious 30–69, malicious from 70) and can combine the email score with an embedded-URL model score. Therefore the saved email-only confusion matrix is not an end-to-end measurement of API predictions.

### Train/inference consistency

The text cleaner is shared by trainer and `/api/check-email`, so the core subject/body text transformation is consistent. The inference route also analyzes links from raw submitted text with the URL model and combines that score; training metrics do not include that second signal. Both trainer and API ignore sender and most MeAJOR structured fields. The API accepts subject and body only (subject max 500 chars; body max 200,000 chars). Submitted email contents are not logged or returned.

The saved artifact metadata says scikit-learn 1.7.2; the read-only inference audit environment has 1.6.1 and emits `InconsistentVersionWarning` while unpickling. Pin and use the declared requirements version for reproducible inference/evaluation.

## 5. False-positive analysis

The saved metrics include only aggregate confusion counts (6,097 false positives) and do not include per-message predictions or false-positive IDs. To provide a content-safe diagnostic, the existing saved model was applied read-only to the current cleaned CSV’s TREC-05 rows at 0.50. It yielded 6,058 benign false positives among 27,352 benign messages (22.15% on this current snapshot), close to but not equal to the historical 6,097 / 22.29%. This diagnostic is not a replacement for or exact reproduction of the saved evaluation.

The following counts are **overlapping keyword/placeholder matches**, not human-verified semantic categories. No full message text is shown.

| Pattern among current-snapshot false positives | Count | Share of 6,058 FPs |
|---|---:|---:|
| Corporate/workplace vocabulary | 1,674 | 27.6% |
| URL placeholder after preprocessing | 1,477 | 24.4% |
| Marketing/promotional vocabulary | 1,163 | 19.2% |
| Account/notification vocabulary | 1,133 | 18.7% |
| Financial/transaction vocabulary | 680 | 11.2% |
| Security/alert/account/password/verification vocabulary | 600 | 9.9% |
| Automated notification vocabulary | 587 | 9.7% |
| Password/login/authentication vocabulary | 364 | 6.0% |
| Urgent/time-pressure vocabulary | 237 | 3.9% |
| Newsletter/unsubscribe vocabulary | 195 | 3.2% |

These broad overlaps support a vocabulary/source-shift explanation: many benign messages use terms that also occur in phishing, especially workplace and transactional language. The model is a bag-of-word unigrams/bigrams classifier, not a semantic or sender-authentication system. URLs are a particularly important limitation: the text model treats every recognized link as the same `URL` token, while API-time URL scoring can add risk separately. The saved confusion matrix does not support claims about exact semantic categories, so the table should be read as a triage guide only.

## 6. MeAJOR structured fields

Availability is measured on the current 108,685-row raw CSV. “API now” refers to the existing `/api/check-email` contract.

| Field | Availability / reliability | Potential value and leakage risk | API now? |
|---|---|---|---|
| Sender | 99.5%; values are anonymized identifiers, not raw addresses | Could group messages for evaluation; sender identity can be a strong shortcut and may not transfer | No |
| Sender domain | 99.5%, 28,464 distinct; likely useful for grouping and reputation context, but domain-label associations are highly source-dependent | High leakage risk if used as a learned categorical feature; an unseen-domain evaluation is necessary | No; could be parsed from a supplied From header |
| Receiver | 97.8%; anonymized identifiers | Useful for grouping; high privacy and recipient-specific leakage risk | No |
| Receiver domain | 97.8%, 6,366 distinct and strongly concentrated | Could capture organizational context, but likely shortcut/recipient leakage | No |
| URL list | Nonempty in 58.9%; 44,633 missing | Raw list can be incomplete; URL count/length/subdomain summaries are populated for all rows but may be derived from the original corpus’s parser | Not as structured fields; route regex-scans visible raw text for scheme/www links and invokes the URL model |
| URL count/statistics | All rows contain numeric values, including zeros | Potentially useful when recomputed consistently at inference; missing-vs-zero semantics and source-specific extraction need validation | Only URLs extracted from submitted text are separately scored; these statistics are not passed to email model |
| Attachment count / flag | 100% populated | Useful only if MIME attachment parsing is available at inference; corpus-specific metadata may be a shortcut | No |
| Attachment types | Only 1.53% populated | Too sparse and likely inconsistent for direct model use | No |
| Content type | 100% except one row; 16 distinct values | MIME structure can help if parsed robustly; collection/source artifacts can leak | No |
| Language | 100% except one row; 243 distinct values, with composite tags | Noisy/duplicated tags make direct use questionable; could become a source shortcut | No |
| Date | 100% populated, but parsed range is 1889–2102 | Current values are not reliable enough as-is. Temporal features could reveal collection/source time rather than phishing behavior | No |
| Source | 100% except one row; three TREC labels | Essential for splitting/reporting, but direct model use would be direct source leakage | No; must never be a prediction feature |

## 7. Proposed Email Model v2 split

The existing TREC-05 final test must remain frozen and must never be used for fitting, calibration, threshold choice, or model selection. Before implementing a split, pin the exact dataset snapshot because the present workspace CSV does not match the saved metrics’ row counts.

For planning against the **current trainer-cleaned CSV**, the following exact row targets are feasible before enforcing indivisible near-duplicate clusters:

| Partition | Proposed rows | Construction |
|---|---:|---|
| Fit/train | 31,466 | TREC-06/07 only; after removing training-pool messages with nonmissing sender domains also present in TREC-05; grouped by sender domain and verified near-duplicate/template cluster |
| Calibration | 6,742 | Separate TREC-06/07 groups, disjoint from fit and validation; fit sigmoid calibration only |
| Validation | 6,743 | Separate TREC-06/07 groups; threshold sweep and model selection only |
| Final test | 46,544 | All current trainer-cleaned TREC-05 rows; frozen and used once for final evaluation |
| Excluded from training/calibration/validation | 12,847 | Current TREC-06/07 rows with sender domains also present in TREC-05 |

The fit/calibration/validation totals (44,951) reflect a 70/15/15 target allocation of the remaining current training pool. Exact source/domain/template-group assignment may shift these row counts because groups must remain intact; the proposed counts are allocation targets, not a completed split. Missing sender domains should not be considered equal to each other. TREC-05’s historical saved metric support is 46,533, not the 46,544 rows in the current cleaned CSV; this discrepancy must be resolved by snapshot pinning before a v2 implementation treats either count as the immutable final test.

Use a deterministic group-aware allocation that balances labels as far as possible while keeping sender domains and near-duplicate/template clusters wholly within one partition. Report per-partition counts and overlap checks. Keep source, sender/receiver identifiers, and test-derived groups out of model features. If strict domain exclusion causes excessive loss or class skew, report that limitation rather than relaxing the final-test boundary invisibly.

## 8. Threshold methodology

Do not assume 0.50. Fit the text model on fit only, fit its calibration map on calibration only, then score validation once. Sweep thresholds over the distinct validation scores (or a fixed 0.01 grid), computing precision, recall, F1, FPR, FNR, and confusion matrix at each threshold.

Pre-register a recall constraint near the current result, for example recall **at least 95%** (and report sensitivity at 96% if feasible). Among thresholds meeting the constraint, select the threshold with the lowest validation FPR; use higher precision and then F1 as tie-breakers. If no threshold meets the recall constraint while materially improving FPR, document the trade-off and do not claim success. Report bootstrap confidence intervals or grouped uncertainty where practical. Freeze model, calibrator, and threshold before evaluating the untouched TREC-05 set once. Never sweep thresholds or choose a model on TREC-05.

## 9. Recommended v2 design

1. Pin and checksum the exact MeAJOR snapshot and reproduce the current training counts from it before comparing models.
2. Keep TREC-05 fixed. Split TREC-06/07 into fit, calibration, and validation groups with sender-domain and verified near-duplicate/template boundaries; remove training domains overlapping TREC-05 for a strict unseen-sender-domain result, and report the row cost.
3. Preserve the lightweight TF-IDF + Logistic Regression baseline for comparison. Evaluate subject/body separately or explicitly weight the subject, and compare word versus character n-grams using only fit/validation. Do not add features based solely on these test false positives.
4. Consider robust URL-derived features only if they can be recomputed identically at training and inference (e.g. URL count, host structure, and whether visible link text differs from destination). Do not reduce all link evidence to URL presence alone.
5. Keep sender/receiver domains, raw sender/receiver IDs, source, and raw date out of the initial predictive feature set. They are useful for grouping and diagnostics but pose direct leakage risks. Add attachment/content-type features only after matching the API’s MIME extraction behavior to the dataset representation.
6. Select a threshold using validation-only metrics and a recall floor; report email-only metrics separately from the API’s combined email-plus-URL output.

## Audit conclusions

- **Most likely causes of high FPR:** source and domain shift; benign corporate, account, transactional, notification, and promotional language overlapping phishing vocabulary; class-balanced lexical training; fixed 0.50 evaluation threshold without validation tuning; and calibration learned from message-random samples that share domains/templates with fit. The saved data does not allow attribution of exact proportions to these causes.
- **Is the evaluation trustworthy?** It is a useful historical TREC-05 source-held-out benchmark and the source boundary is respected in saved metadata. It is not an unseen-domain/template evaluation, is not an end-to-end API evaluation, and is not exactly reproducible from the current workspace CSV. Treat the reported metrics as provisional until data snapshot and sklearn version are pinned.
- **Features to consider later:** consistently extracted subject/body representations; URL count/host structure; carefully parsed MIME attachment/content-type data if available in the API; sender domain only for grouping initially, and as a predictive feature only after a strict unseen-domain evaluation.
- **Features not to use initially:** source, raw sender/receiver identifiers, receiver domain, raw date, current sparse attachment-type field, language tags without normalization, or domain allow/deny rules. These can encode source, campaign, recipient, or dataset collection shortcuts.
- **Next implementation step:** make an immutable, checksummed copy/reference of the exact current MeAJOR snapshot and add a read-only split/duplicate audit manifest with row IDs or hashes (never message contents); then implement the grouped fit/calibration/validation split and threshold sweep without touching TREC-05. Train Model v2 only after that split is reviewed and authorized.
