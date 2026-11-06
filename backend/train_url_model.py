from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from domain_utils import registrable_domain
from prepare_url_dataset import DEFAULT_HARD_NEGATIVES, DEFAULT_PREPARED_HARD_NEGATIVES, DEFAULT_UCI, prepare
from risk_engine import url_risk_decision
from url_features import (FEATURE_NAMES, canonicalize_root_path_for_model,
                           extract_url_features, normalize_url, url_indicators,
                           url_signal_count)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MODEL_DIR = ROOT / "models"
DEFAULT_PREPARED = PROJECT_ROOT / "data/processed/url_training.csv"
SEED = 42
THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.30, 1.00, 0.01))
HARD_NEGATIVE_VALIDATION_FPR_CAP = 0.10


def _read_data(path: Path, source: str) -> tuple[pd.DataFrame, dict]:
    if source == "uci":
        _, stats = prepare(DEFAULT_UCI, DEFAULT_PREPARED, None, DEFAULT_HARD_NEGATIVES)
        path = DEFAULT_PREPARED
    elif source == "uci-csv":
        if not path:
            raise ValueError("--source uci-csv requires --csv")
        _, stats = prepare(path, DEFAULT_PREPARED, None, DEFAULT_HARD_NEGATIVES)
        path = DEFAULT_PREPARED
        source = "prepared"
    elif source == "prepared":
        stats_path = path.with_name("url_dataset_stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    else:
        stats = {}
    frame = pd.read_csv(path, low_memory=False)
    columns = {str(column).lower(): column for column in frame.columns}
    url_col = columns.get("url")
    label_col = columns.get("label") or columns.get("target") or columns.get("class")
    if not url_col or not label_col:
        raise ValueError("Training CSV requires URL and label columns")
    labels = pd.to_numeric(frame[label_col], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Project labels must be binary: 0=legitimate, 1=phishing")
    if source == "uci-csv":
        labels = 1 - labels.astype(int)
    result = pd.DataFrame({
        "url": frame[url_col], "label": labels.astype(int),
        "source": frame[columns["source"]].fillna("unspecified").astype(str) if "source" in columns else source,
        "observed_at": frame[columns["observed_at"]] if "observed_at" in columns else "",
    })
    return result, stats


def _splits(y: np.ndarray, groups: np.ndarray, dates: pd.Series) -> tuple[dict[str, np.ndarray], dict]:
    timestamp = pd.to_datetime(dates, utc=True, errors="coerce")
    if len(timestamp) and timestamp.notna().all() and timestamp.nunique() >= 20:
        # Time bins are assigned first; any registrable domain spanning bins is
        # purged completely, preserving both chronology and domain isolation.
        q = timestamp.quantile([0.60, 0.70, 0.80]).tolist()
        bins = np.digitize(timestamp.astype("int64"), np.asarray([x.value for x in q]), right=True)
        domains_by_bin = [set(groups[bins == i]) for i in range(4)]
        cross_bin = set()
        for left in range(4):
            for right in range(left + 1, 4):
                cross_bin.update(domains_by_bin[left] & domains_by_bin[right])
        keep = ~np.isin(groups, list(cross_bin))
        split = {name: np.flatnonzero(keep & (bins == i)) for i, name in enumerate(("fit", "calibration", "validation", "test"))}
        if all(len(np.unique(y[idx])) == 2 for idx in split.values()):
            return split, {"method": "chronological 60/10/10/20 row-time bins with all cross-bin registrable domains purged", "cutoffs_utc": [x.isoformat() for x in q], "purged_cross_bin_domains": len(cross_bin), "temporal": True}

    # The checked-in PhiUSIIL corpus has no observation timestamps. Split by
    # registrable domain into disjoint folds; no row-random fallback is used.
    for seed in range(SEED, SEED + 30):
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_ids = [test_idx for _, test_idx in splitter.split(np.zeros(len(y)), y, groups)]
        split = {
            "fit": np.concatenate(fold_ids[:6]),
            "calibration": fold_ids[6],
            "validation": fold_ids[7],
            "test": np.concatenate(fold_ids[8:]),
        }
        if all(len(np.unique(y[idx])) == 2 for idx in split.values()):
            domain_sets = {name: set(groups[idx]) for name, idx in split.items()}
            if all(not (domain_sets[a] & domain_sets[b]) for a in split for b in split if a < b):
                return split, {"method": "registrable-domain-disjoint StratifiedGroupKFold; 60/10/10/20 folds", "seed": seed, "temporal": False, "timestamp_limitation": "source has no per-URL observation time"}
    raise ValueError("Could not create four domain-disjoint partitions containing both labels")


def _models() -> dict:
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=100_000, sublinear_tf=True)
    char_no_generic = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_df=0.05, max_features=100_000, sublinear_tf=True)
    word = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)[a-zA-Z0-9_]+", ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True)
    word_no_generic = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)[a-zA-Z0-9_]+", ngram_range=(1, 2), min_df=2, max_df=0.10, max_features=60_000, sublinear_tf=True)
    numeric = ColumnTransformer([("numeric", StandardScaler(), FEATURE_NAMES)], remainder="drop")
    char_only = ColumnTransformer([("char", char, "url")], remainder="drop")
    word_char = ColumnTransformer([("word", word, "url"), ("char", char, "url")], remainder="drop")
    word_char_no_generic = ColumnTransformer([("word", word_no_generic, "url"), ("char", char_no_generic, "url")], remainder="drop")
    combined = ColumnTransformer([("char", char, "url"), ("numeric", StandardScaler(), FEATURE_NAMES)], remainder="drop")
    lr = lambda: LogisticRegression(C=2.0, max_iter=1500, class_weight="balanced", random_state=SEED)
    return {
        "char_tfidf_logistic_regression": make_pipeline(char_only, lr()),
        "char_tfidf_no_generic_logistic_regression": make_pipeline(ColumnTransformer([("char", char_no_generic, "url")], remainder="drop"), lr()),
        "word_char_tfidf_logistic_regression": make_pipeline(word_char, lr()),
        "word_char_no_generic_logistic_regression": make_pipeline(word_char_no_generic, lr()),
        "char_tfidf_linear_svm": make_pipeline(char_only, LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        "numeric_random_forest": make_pipeline(numeric, RandomForestClassifier(n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        "numeric_hist_gradient_boosting": make_pipeline(numeric, HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, random_state=SEED)),
        "combined_char_numeric_logistic_regression": make_pipeline(combined, lr()),
    }


def _input(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    if name.startswith("numeric_"):
        return frame[FEATURE_NAMES]
    return frame


def _probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(X)[:, classes.index(1)]
    raise TypeError("Calibrated candidate must expose predict_proba")


def _metric_row(y_true, probability: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "sample_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None,
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"legitimate": int((y_true == 0).sum()), "phishing": int((y_true == 1).sum())},
    }


def _calibrate(base, name, X_cal, y_cal, X_val, y_val, prefer_smooth: bool = False) -> tuple[object, dict, str]:
    baseline = None
    if hasattr(base, "predict_proba"):
        baseline = base.predict_proba(_input(name, X_val))[:, list(base.classes_).index(1)]
        baseline_brier = float(brier_score_loss(y_val, baseline))
    else:
        baseline_brier = None
    candidates = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        calibrated.fit(_input(name, X_cal), y_cal)
        probability = _probability(calibrated, _input(name, X_val))
        candidates[method] = (calibrated, probability, float(brier_score_loss(y_val, probability)))
    # The prior v3 audit showed isotonic calibration saturating out-of-domain
    # URL scores at 1.0. Prefer the smoother sigmoid map for the v4 lexical
    # models; numeric models still select calibration by validation Brier.
    selected_method = "sigmoid" if prefer_smooth else min(candidates, key=lambda method: candidates[method][2])
    selected_model, selected_probability, selected_brier = candidates[selected_method]
    if baseline_brier is not None and selected_brier >= baseline_brier:
        return base, {"method": "none", "validation_brier": baseline_brier, "sigmoid_validation_brier": candidates["sigmoid"][2], "isotonic_validation_brier": candidates["isotonic"][2], "improved_over_uncalibrated": False}, baseline
    return selected_model, {"method": selected_method, "validation_brier": selected_brier, "baseline_validation_brier": baseline_brier, "sigmoid_validation_brier": candidates["sigmoid"][2], "isotonic_validation_brier": candidates["isotonic"][2], "improved_over_uncalibrated": selected_brier < baseline_brier if baseline_brier is not None else None}, selected_probability


def _threshold_results(y_val: np.ndarray, probability: np.ndarray, hard_negative_probability: np.ndarray | None = None) -> tuple[list[dict], float]:
    rows = [_metric_row(y_val, probability, threshold) for threshold in THRESHOLDS]
    for row in rows:
        row["hard_negative_validation_fpr"] = float(np.mean(hard_negative_probability >= row["threshold"])) if hard_negative_probability is not None and len(hard_negative_probability) else None
    eligible = [row for row in rows if row["false_positive_rate"] is not None and row["false_positive_rate"] <= 0.02 and (row["hard_negative_validation_fpr"] is None or row["hard_negative_validation_fpr"] <= HARD_NEGATIVE_VALIDATION_FPR_CAP)]
    if eligible:
        # Maximize recall under both validation FPR caps.
        chosen = max(eligible, key=lambda row: (row["recall"], -(row["hard_negative_validation_fpr"] or 0.0), row["f1"], row["threshold"]))
    else:
        broad_fpr_eligible = [row for row in rows if row["false_positive_rate"] is not None and row["false_positive_rate"] <= 0.02]
        if broad_fpr_eligible and hard_negative_probability is not None and len(hard_negative_probability):
            chosen = min(broad_fpr_eligible, key=lambda row: (row["hard_negative_validation_fpr"], -row["recall"], -row["threshold"]))
        else:
            chosen = min(rows, key=lambda row: (row["false_positive_rate"] + (row["hard_negative_validation_fpr"] or 0.0), -row["recall"], -row["threshold"]))
    return rows, float(chosen["threshold"])


HARD_NEGATIVE_SAMPLE_WEIGHT = 10.0


def _fit_with_hard_negative_weights(estimator, model_name: str, X: pd.DataFrame, y: np.ndarray, hard_negative_count: int) -> None:
    weights = np.concatenate([np.ones(len(y) - hard_negative_count), np.full(hard_negative_count, HARD_NEGATIVE_SAMPLE_WEIGHT)])
    final_step_name = estimator.steps[-1][0]
    estimator.fit(_input(model_name, X), y, **{f"{final_step_name}__sample_weight": weights})


def _latency_and_size(model, name: str, examples: list[str]) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        joblib.dump(model, path, compress=3)
        size = path.stat().st_size
    finally:
        path.unlink(missing_ok=True)
    durations = []
    for url in examples:
        features = extract_url_features(url)
        row = pd.DataFrame([[features[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)
        if not name.startswith("numeric_"):
            row.insert(0, "url", url)
        started = time.perf_counter()
        _probability(model, _input(name, row))
        durations.append((time.perf_counter() - started) * 1000)
    return {"model_size_bytes_compressed": int(size), "inference_latency_ms_mean": float(np.mean(durations)), "inference_latency_ms_p95": float(np.percentile(durations, 95)), "latency_sample_count": len(durations), "latency_includes": "URL feature extraction and single-row model prediction"}


def _prepare_hard_negative_partitions(path: Path, improved: bool = False, known_domains: set[str] | None = None, canonicalize_root_slash: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    frame = pd.read_csv(path)
    records = []
    for record in frame.itertuples(index=False):
        url = normalize_url(str(record.url))
        if canonicalize_root_slash:
            url = canonicalize_root_path_for_model(url)
        domain = registrable_domain(urlsplit(url).hostname or "")
        source = str(getattr(record, "source", "manual curated challenge"))
        records.append({"url": url, "domain": domain, "category": str(record.category), "expected_label": int(record.expected_label), "source": source})
    frame = pd.DataFrame(records).drop_duplicates("url").reset_index(drop=True)
    if improved:
        # Old hand-curated examples are development data after the v3 review;
        # UCI mined examples augment fit only. New official roots alone form
        # validation and final challenge partitions, all domain-disjoint.
        curated = pd.read_csv(DEFAULT_HARD_NEGATIVES)
        curated_domains = {registrable_domain(urlsplit(normalize_url(str(url))).hostname or "") for url in curated.url}
        uci_domains = set(frame.loc[frame.source.eq("UCI PhiUSIIL 967"), "domain"])
        if known_domains:
            uci_domains.update(known_domains)
        external_domains = np.asarray(sorted(set(frame.loc[frame.source.eq("external official URL challenge"), "domain"]) - curated_domains - uci_domains))
        if len(external_domains) < 20:
            raise ValueError("Expanded external challenge needs at least 20 distinct registrable domains")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + 20)
        remaining_idx, test_idx = next(splitter.split(external_domains, groups=external_domains))
        remaining_domains = external_domains[remaining_idx]
        test_domains = set(external_domains[test_idx])
        val_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 21)
        train_idx, validation_idx = next(val_split.split(remaining_domains, groups=remaining_domains))
        train_domains = curated_domains | uci_domains | set(remaining_domains[train_idx])
        validation_domains = set(remaining_domains[validation_idx])
        return (
            frame[frame.domain.isin(train_domains)].reset_index(drop=True),
            frame[frame.domain.isin(validation_domains)].reset_index(drop=True),
            frame[frame.domain.isin(test_domains)].reset_index(drop=True),
            set(frame.domain),
        )
    domains = np.asarray(sorted(frame.domain.unique()))
    _, random_test = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(domains, groups=domains))
    test_domains = set(domains[random_test])
    # Keep the exact concerns named in the request in the final challenge holdout.
    # This only determines evaluation membership; no domain gets a model rule.
    reserved_test_domains = {"google.com", "amazon.in", "microsoft.com", "github.com", "apple.com", "paypal.com", "login.gov", "bvuniversity.edu.in", "bankofamerica.com"}
    test_domains.update(reserved_test_domains & set(domains))
    remaining = np.asarray(sorted(set(domains) - test_domains))
    _, validation_ids = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1).split(remaining, groups=remaining))
    validation_domains = set(remaining[validation_ids])
    training_domains = set(remaining) - validation_domains
    return (
        frame[frame.domain.isin(training_domains)].reset_index(drop=True),
        frame[frame.domain.isin(validation_domains)].reset_index(drop=True),
        frame[frame.domain.isin(test_domains)].reset_index(drop=True),
        set(frame.domain),
    )


def _evaluate_hard_negatives(frame: pd.DataFrame, model, model_name: str, thresholds: dict, train_domains: set[str] | None, canonicalize_root_slash: bool = False) -> dict:
    rows = []
    for _, record in frame.iterrows():
        url = normalize_url(str(record.url))
        if canonicalize_root_slash:
            url = canonicalize_root_path_for_model(url)
        features = extract_url_features(url)
        row = pd.DataFrame([[features[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)
        if not model_name.startswith("numeric_"):
            row.insert(0, "url", url)
        probability = float(_probability(model, _input(model_name, row))[0])
        group = registrable_domain(urlsplit(url).hostname or "")
        prediction, score = url_risk_decision(probability, thresholds, url_signal_count(features))
        rows.append({"url": url, "category": str(record.category), "expected_label": int(record.expected_label), "model_probability": probability, "prediction": prediction, "risk_score": score, "domain_seen_in_fit": (group in train_domains) if train_domains is not None else None, "indicators": url_indicators(features, url)})
    return _group_metrics(rows, thresholds["suspicious_probability"])


def _hard_negative_probabilities(frame: pd.DataFrame, model, model_name: str, canonicalize_root_slash: bool = False) -> np.ndarray:
    probabilities = []
    for url in frame.url.astype(str):
        normalized = normalize_url(url)
        if canonicalize_root_slash:
            normalized = canonicalize_root_path_for_model(normalized)
        features = extract_url_features(normalized)
        row = pd.DataFrame([[features[col] for col in FEATURE_NAMES]], columns=FEATURE_NAMES)
        if not model_name.startswith("numeric_"):
            row.insert(0, "url", normalized)
        probabilities.append(float(_probability(model, _input(model_name, row))[0]))
    return np.asarray(probabilities, dtype=float)


def _group_metrics(rows: list[dict], threshold: float = 0.70) -> dict:
    if not rows:
        return {"sample_count": 0}
    y = np.asarray([row["expected_label"] for row in rows], dtype=int)
    p = np.asarray([row["model_probability"] for row in rows], dtype=float)
    thresholded = [{**row, "threshold_prediction": int(p[i] >= threshold)} for i, row in enumerate(rows)]
    result = _metric_row(y, p, threshold)
    result["model_warning_fpr_at_selected_threshold"] = float(sum(y[i] == 0 and p[i] >= threshold for i in range(len(y))) / max(1, sum(y == 0)))
    result["results"] = thresholded
    return result


def _error_analysis(rows: list[dict], features_by_url: dict[str, dict], hard_negative_rows: list[dict]) -> str:
    fp = [row for row in rows if row["expected_label"] == 0 and row["threshold_prediction"] == 1]
    fn = [row for row in rows if row["expected_label"] == 1 and row["threshold_prediction"] == 0]
    def examples(items):
        return "\n".join(f"- `{r['url']}` — p={r['model_probability']:.4f}; features: {', '.join(r['indicators']) or 'no listed indicator'}" for r in items[:15]) or "- None at the selected threshold."
    common_fp = {}
    for row in fp:
        features = features_by_url[row["url"]]
        for key in ("has_unusual_port", "has_punycode", "is_ip_host", "suspicious_host_word_count", "suspicious_path_word_count", "query_param_count", "url_length"):
            common_fp[key] = common_fp.get(key, 0) + int(bool(features.get(key)))
    common_pattern_text = ", ".join(f"{name} in {count}/{len(fp)} false positives" for name, count in common_fp.items() if count and name != "url_length") or "No repeated listed feature was shared by false positives"
    hard_flagged = [row for row in hard_negative_rows if row.get("threshold_prediction") == 1]
    return ("# URL error analysis\n\n"
        f"Selected threshold: {CURRENT_THRESHOLD:.2f}. Errors are from the registrable-domain-disjoint UCI test partition, after model selection.\n\n"
        f"False positives: {len(fp)}.\n\n{examples(fp)}\n\n"
        f"False negatives: {len(fn)}.\n\n{examples(fn)}\n\n"
        f"Common false-positive feature presence: {common_pattern_text}.\n\n"
        f"Hard-negative challenge false positives: {CURRENT_HARD_FP} of {CURRENT_HARD_N} ({(100*CURRENT_HARD_FP/max(1,CURRENT_HARD_N)):.1f}%).\n\n"
        "## Hard-negative examples flagged\n\n"
        + ("\n".join(f"- `{row['url']}` — {row['category']}; p={row['model_probability']:.4f}" for row in hard_flagged[:20]) or "- None at the selected threshold.")
        + "\n\n"
        "## Weaknesses and next steps\n\n"
        "This URL-only model cannot observe page content, destination reputation, redirects, or when a URL is newly registered. Current UCI data has no timestamps, so this is not a temporal evaluation. The hard-negative set is small and mixes manually curated URLs with URLs mined from the UCI legitimate class; report it separately and expand it with independently verified examples. Check the listed false positives and false negatives before changing any feature or threshold. Do not add domain-specific allow rules.\n")


CURRENT_THRESHOLD = 0.70
CURRENT_HARD_FP = 0
CURRENT_HARD_N = 0


def train(csv_path: Path, hard_negatives: Path, run_standard: bool = False, candidate_version: int = 3) -> dict:
    canonicalize_root_slash = candidate_version >= 6
    frame, prep_stats = _read_data(csv_path, "prepared")
    improved_candidate = candidate_version >= 4
    prepared_domains = {
        registrable_domain(urlsplit(normalize_url(str(url))).hostname or "")
        for url in frame.url.astype(str)
    } if improved_candidate else None
    hard_train, hard_validation, hard_test, hard_all_domains = _prepare_hard_negative_partitions(
        hard_negatives, improved=improved_candidate, known_domains=prepared_domains,
        canonicalize_root_slash=canonicalize_root_slash,
    )
    raw_count = len(frame)
    clean_rows, rejected, conflicting = [], 0, 0
    for row in frame.itertuples(index=False):
        try:
            url = normalize_url(str(row.url))
            if canonicalize_root_slash:
                url = canonicalize_root_path_for_model(url)
            label = int(row.label)
        except (ValueError, TypeError):
            rejected += 1
            continue
        clean_rows.append({"url": url, "label": label, "source": str(row.source), "observed_at": str(row.observed_at or "")})
    data = pd.DataFrame(clean_rows)
    conflicts = set(data.groupby("url").label.nunique().loc[lambda values: values > 1].index)
    conflicting = len(conflicts)
    data = data[~data.url.isin(conflicts)].drop_duplicates("url").reset_index(drop=True)
    data["domain"] = [registrable_domain(urlsplit(url).hostname or "") for url in data.url]
    if improved_candidate:
        hard_partition_domains = [set(part.domain) for part in (hard_train, hard_validation, hard_test)]
        if any(hard_partition_domains[a] & hard_partition_domains[b] for a in range(3) for b in range(a + 1, 3)):
            raise RuntimeError("Hard-negative registrable domains cross fit, validation, and final challenge")
        historical = pd.read_csv(DEFAULT_HARD_NEGATIVES)
        historical_domains = {registrable_domain(urlsplit(normalize_url(str(url))).hostname or "") for url in historical.url}
        if set(hard_test.domain) & set(data.domain) or set(hard_test.domain) & historical_domains:
            raise RuntimeError("Fresh hard-negative final roots overlap UCI or the earlier development challenge")
    # Keep every curated challenge root out of the broad corpus so all hard
    # negative roots remain independent of fit/calibration/validation/test.
    hard_negative_overlap_rows = int(data.domain.isin(hard_all_domains).sum())
    data = data[~data.domain.isin(hard_all_domains)].reset_index(drop=True)
    valid = data.domain.ne("")
    rejected += int((~valid).sum())
    data = data[valid].reset_index(drop=True)
    if data.label.nunique() != 2:
        raise ValueError("Prepared URL data needs both project labels (0=legitimate, 1=phishing)")
    feature_rows, valid_indices = [], []
    for index, url in enumerate(data.url):
        try:
            feature_rows.append(extract_url_features(url))
            valid_indices.append(index)
        except ValueError:
            rejected += 1
    data = data.iloc[valid_indices].reset_index(drop=True)
    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    features.insert(0, "url", data.url.astype(str))
    labels = data.label.to_numpy(dtype=int)
    groups = data.domain.to_numpy()
    splits, split_info = _splits(labels, groups, data.observed_at)
    for left, left_ids in splits.items():
        for right, right_ids in splits.items():
            if left < right and set(groups[left_ids]) & set(groups[right_ids]):
                raise RuntimeError(f"Registrable-domain leakage between {left} and {right}")
    split_info["rows"] = {key: int(len(value)) for key, value in splits.items()}
    split_info["fit_rows_including_hard_negative_training"] = int(len(splits["fit"]) + len(hard_train))
    split_info["hard_negative_train_rows"] = int(len(hard_train))
    split_info["hard_negative_train_domains"] = int(hard_train.domain.nunique())
    split_info["hard_negative_validation_rows"] = int(len(hard_validation))
    split_info["hard_negative_validation_domains"] = int(hard_validation.domain.nunique())
    split_info["hard_negative_test_rows"] = int(len(hard_test))
    split_info["hard_negative_test_domains"] = int(hard_test.domain.nunique())
    split_info["hard_negative_domains_removed_from_UCI"] = hard_negative_overlap_rows
    if improved_candidate:
        split_info["hard_negative_final_roots_overlap_UCI"] = 0
        split_info["hard_negative_final_roots_overlap_prior_development_set"] = 0
    split_info["registrable_domains"] = {key: int(len(set(groups[value]))) for key, value in splits.items()}
    split_info["domain_overlap_counts"] = {f"{a}_{b}": 0 for a in splits for b in splits if a < b}
    split_info["class_counts"] = {key: {"legitimate": int((labels[value] == 0).sum()), "phishing": int((labels[value] == 1).sum())} for key, value in splits.items()}

    y_fit, y_cal, y_val = (labels[splits[key]] for key in ("fit", "calibration", "validation"))
    X_fit, X_cal, X_val = (features.iloc[splits[key]] for key in ("fit", "calibration", "validation"))
    hard_features = [extract_url_features(url) for url in hard_train.url]
    X_hard = pd.DataFrame(hard_features, columns=FEATURE_NAMES)
    X_hard.insert(0, "url", hard_train.url.astype(str).to_numpy())
    X_fit = pd.concat([X_fit, X_hard], ignore_index=True)
    y_fit = np.concatenate([y_fit, np.zeros(len(hard_train), dtype=int)])
    hard_validation_features = [extract_url_features(url) for url in hard_validation.url]
    X_hard_validation = pd.DataFrame(hard_validation_features, columns=FEATURE_NAMES)
    X_hard_validation.insert(0, "url", hard_validation.url.astype(str).to_numpy())
    comparisons, fitted, probs_val, thresholds_by_model = {}, {}, {}, {}
    examples_for_latency = data.url.iloc[splits["validation"]].head(100).tolist()
    for model_name, estimator in _models().items():
        fit_started = time.perf_counter()
        _fit_with_hard_negative_weights(estimator, model_name, X_fit, y_fit, len(hard_train))
        fit_seconds = time.perf_counter() - fit_started
        fitted[model_name] = estimator
        calibration_started = time.perf_counter()
        calibrated, calibration_info, val_probability = _calibrate(
            estimator, model_name, X_cal, y_cal, X_val, y_val,
            prefer_smooth=improved_candidate and not model_name.startswith("numeric_"),
        )
        calibration_seconds = time.perf_counter() - calibration_started
        if isinstance(val_probability, str):
            val_probability = _probability(calibrated, _input(model_name, X_val))
        hard_val_probability = _probability(calibrated, _input(model_name, X_hard_validation)) if len(X_hard_validation) else np.asarray([])
        rows, selected_threshold = _threshold_results(y_val, val_probability, hard_val_probability)
        probs_val[model_name] = val_probability
        thresholds_by_model[model_name] = selected_threshold
        latency = _latency_and_size(calibrated, model_name, examples_for_latency[:30])
        comparisons[model_name] = {
            "validation_metrics_at_0_50": _metric_row(y_val, val_probability, 0.50),
            "threshold_sweep": rows,
            "selected_validation_threshold": selected_threshold,
            "selection_rule": f"within this model, maximize recall under UCI validation FPR <= 0.02 and hard-negative validation FPR <= {HARD_NEGATIVE_VALIDATION_FPR_CAP:.0%}; if infeasible, minimize hard-negative validation FPR subject to UCI validation FPR <= 0.02, then maximize recall",
            "calibration": calibration_info,
            "hard_negative_validation_fpr_at_selected_threshold": next(row["hard_negative_validation_fpr"] for row in rows if row["threshold"] == selected_threshold),
            "fit_seconds": round(fit_seconds, 3), "calibration_and_validation_seconds": round(calibration_seconds, 3),
            **latency,
        }
        fitted[model_name] = calibrated
        print(f"Compared {model_name}", flush=True)

    eligible = [name for name, result in comparisons.items() if any(row["threshold"] == result["selected_validation_threshold"] and row["false_positive_rate"] <= 0.02 and (row["hard_negative_validation_fpr"] is None or row["hard_negative_validation_fpr"] <= HARD_NEGATIVE_VALIDATION_FPR_CAP) for row in result["threshold_sweep"])]
    if eligible:
        selected_name = max(eligible, key=lambda name: (comparisons[name]["threshold_sweep"][THRESHOLDS.index(thresholds_by_model[name])]["recall"], -(comparisons[name]["hard_negative_validation_fpr_at_selected_threshold"] or 0.0), comparisons[name]["validation_metrics_at_0_50"]["pr_auc"] or 0.0, comparisons[name]["validation_metrics_at_0_50"]["f1"]))
        selection_rule = f"Highest phishing recall subject to UCI validation FPR <= 2% and hard-negative validation FPR <= {HARD_NEGATIVE_VALIDATION_FPR_CAP:.0%}; ties prefer lower hard-negative FPR, then PR-AUC and F1. Both final test partitions are excluded from selection."
    else:
        selected_name = min(comparisons, key=lambda name: (comparisons[name]["hard_negative_validation_fpr_at_selected_threshold"] if comparisons[name]["hard_negative_validation_fpr_at_selected_threshold"] is not None else 1.0, -comparisons[name]["threshold_sweep"][THRESHOLDS.index(thresholds_by_model[name])]["recall"], -(comparisons[name]["validation_metrics_at_0_50"]["pr_auc"] or 0.0), -comparisons[name]["validation_metrics_at_0_50"]["f1"]))
        selection_rule = "No candidate met both validation FPR caps; selected the model/threshold with the lowest hard-negative validation FPR among thresholds meeting UCI validation FPR <= 2%, then highest phishing recall. Final tests were excluded from selection."
    global CURRENT_THRESHOLD
    CURRENT_THRESHOLD = thresholds_by_model[selected_name]
    model = fitted[selected_name]
    selected_thresholds = {"suspicious_probability": CURRENT_THRESHOLD, "malicious_probability": max(CURRENT_THRESHOLD, 0.90), "selection_rule": selection_rule, "structural_severity_guard": "MALICIOUS requires two independent structural signals; otherwise high probability remains SUSPICIOUS."}

    # Final domain-disjoint test is used once, after model, calibration, and
    # operating threshold selection are complete.
    test_idx = splits["test"]
    test_probability = _probability(model, _input(selected_name, features.iloc[test_idx]))
    test_metrics = _metric_row(labels[test_idx], test_probability, CURRENT_THRESHOLD)
    test_metrics["threshold_sweep"] = [_metric_row(labels[test_idx], test_probability, t) for t in THRESHOLDS]
    test_metrics["probability_mean"] = float(np.mean(test_probability))
    test_metrics["evaluation_group"] = "registrable-domain-disjoint unseen domains"
    version_comparisons = {}
    fresh_candidate_test_metrics = {}
    for candidate_name, candidate_model in fitted.items():
        candidate_probability = _probability(candidate_model, _input(candidate_name, features.iloc[test_idx]))
        candidate_metric = _metric_row(labels[test_idx], candidate_probability, thresholds_by_model[candidate_name])
        candidate_metric["threshold_selected_on_validation"] = thresholds_by_model[candidate_name]
        candidate_metric["threshold_sweep"] = [_metric_row(labels[test_idx], candidate_probability, t) for t in THRESHOLDS]
        fresh_candidate_test_metrics[candidate_name] = candidate_metric
    baseline_comparison = {"status": "not_available"}
    baseline_hard_metrics = {"status": "not_available"}
    validation_idx = splits["validation"]
    legacy_versions = [
        ("v2", MODEL_DIR / "url_model.joblib"),
        ("v3", MODEL_DIR / "url_model_v3.joblib"),
    ]
    if candidate_version >= 5:
        legacy_versions.append(("v4", MODEL_DIR / "url_model_v4.joblib"))
    if candidate_version >= 6:
        legacy_versions.append(("v5", MODEL_DIR / "url_model_v5.joblib"))
    for version, path in legacy_versions:
        if not path.is_file():
            version_comparisons[version] = {"status": "artifact_not_available"}
            continue
        old_artifact = joblib.load(path)
        old_model = old_artifact["model"]
        old_name = str(old_artifact.get("model_name", version))
        old_threshold = float(old_artifact.get("decision_thresholds", {}).get("suspicious_probability", 0.70))
        old_test_probability = _probability(old_model, _input(old_name, features.iloc[test_idx]))
        old_test_metrics = _metric_row(labels[test_idx], old_test_probability, old_threshold)
        old_hard_metrics = _evaluate_hard_negatives(
            hard_test, old_model, old_name,
            {"suspicious_probability": old_threshold, "malicious_probability": max(old_threshold, 0.90)}, None,
            canonicalize_root_slash=canonicalize_root_slash,
        )
        old_latency = _latency_and_size(old_model, old_name, examples_for_latency[:30])
        version_comparisons[version] = {
            "model_name": old_name,
            "threshold_used": old_threshold,
            "overall_test_metrics": old_test_metrics,
            "hard_negative_test_metrics": old_hard_metrics,
            "latency_and_size": old_latency,
            "note": "Legacy artifact scored with its previously selected threshold on v4's final rows. The old artifact may have seen some UCI rows/domains during its original fit, so its UCI test metrics are retrospective, not a fresh holdout. External challenge domains are absent from UCI and the prior hard-negative list.",
        }
        if version == "v2":
            baseline_comparison = {"model_name": old_name, "same_test_at_candidate_selected_threshold": old_test_metrics, "note": version_comparisons[version]["note"]}
            baseline_hard_metrics = old_hard_metrics
    test_rows = []
    test_features = {}
    for idx, probability in zip(test_idx, test_probability):
        url = data.url.iloc[idx]
        row = {"url": url, "expected_label": int(labels[idx]), "model_probability": float(probability), "threshold_prediction": int(probability >= CURRENT_THRESHOLD), "indicators": url_indicators(features.iloc[idx][FEATURE_NAMES].to_dict(), url)}
        test_rows.append(row)
        test_features[url] = features.iloc[idx][FEATURE_NAMES].to_dict()

    global CURRENT_HARD_FP, CURRENT_HARD_N
    hard_metrics = _evaluate_hard_negatives(
        hard_test, model, selected_name, selected_thresholds,
        set(groups[splits["fit"]]) | set(hard_train.domain),
        canonicalize_root_slash=canonicalize_root_slash,
    )
    hard_rows = hard_metrics.get("results", [])
    CURRENT_HARD_N = int(hard_metrics.get("support", {}).get("legitimate", 0))
    CURRENT_HARD_FP = sum(row["model_probability"] >= CURRENT_THRESHOLD for row in hard_rows if row["expected_label"] == 0)
    hard_categories = {}
    for category in sorted({row["category"] for row in hard_rows}):
        subset = [row for row in hard_rows if row["category"] == category]
        hard_categories[category] = _group_metrics(subset, CURRENT_THRESHOLD)
    hard_metrics["categories"] = hard_categories
    hard_metrics["threshold_used"] = CURRENT_THRESHOLD
    hard_metrics["note"] = ("Fresh held-out external official URLs on registrable roots absent from UCI and the earlier hard-negative set; descriptive only and not used for fit or threshold/model selection."
                             if improved_candidate else "Held-out challenge URLs are manually curated or mined from UCI legitimate rows; descriptive only, not used for fit or threshold/model selection.")
    hard_validation_metrics = _evaluate_hard_negatives(
        hard_validation, model, selected_name, selected_thresholds,
        set(groups[splits["fit"]]) | set(hard_train.domain),
        canonicalize_root_slash=canonicalize_root_slash,
    )
    version_comparisons[f"v{candidate_version}"] = {
        "model_name": selected_name,
        "threshold_used": CURRENT_THRESHOLD,
        "overall_test_metrics": test_metrics,
        "hard_negative_test_metrics": hard_metrics,
        "latency_and_size": comparisons[selected_name],
        "note": "New candidate; model and threshold selected without using either final test partition.",
    }

    recent = {"status": "not_available", "reason": "Training corpus has no per-URL observation timestamps; no recent-URL claim is made."}
    evaluation = {
        "general_unseen_urls": test_metrics,
        "unseen_domains": test_metrics,
        "recent_urls": recent,
        "hard_negative_legitimate_urls": hard_metrics,
        "suspicious_looking_legitimate_urls": {key: value for key, value in hard_categories.items() if "suspicious" in key.lower() or "login" in key.lower() or "payment" in key.lower() or "account" in key.lower()},
        "clearly_malicious_phishing_urls": {"sample_count": test_metrics["support"]["phishing"], "metrics": _metric_row(labels[test_idx][labels[test_idx] == 1], test_probability[labels[test_idx] == 1], CURRENT_THRESHOLD), "note": "Positive-only slice; false-positive rate is undefined for this category."},
    }

    now = datetime.now(timezone.utc).isoformat()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = MODEL_DIR / f"url_model_v{candidate_version}.joblib"
    version_tag = f"v{candidate_version}"
    candidate_release = f"{candidate_version}.0.0-candidate"
    source_names = sorted(set(data.source.astype(str)))
    artifact = {
        "model": model, "feature_names": FEATURE_NAMES, "uses_raw_url_text": not selected_name.startswith("numeric_"),
        "canonicalize_root_slash": canonicalize_root_slash,
        "risk_classes": {"malicious": 1}, "training_source": "real", "model_name": selected_name,
        "model_version": candidate_release, "probability_calibrated": comparisons[selected_name]["calibration"]["method"] != "none",
        "calibration_method": comparisons[selected_name]["calibration"]["method"], "decision_thresholds": selected_thresholds,
        "candidate_status": "not_promoted; default REAL artifact remains unchanged",
    }
    joblib.dump(artifact, candidate_path, compress=3)
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            peak_rss *= 1024
    except (AttributeError, OSError):
        peak_rss = None

    inspection = {
        "raw_rows": raw_count, "prepared_rows": len(data), "rejected_rows": rejected,
        "conflicting_label_urls_removed": conflicting,
        "class_counts": {"legitimate": int((labels == 0).sum()), "phishing": int((labels == 1).sum())},
        "unique_registrable_domains": int(pd.Series(groups).nunique()), "source_counts": {str(k): int(v) for k, v in data.source.value_counts().items()},
        "hard_negative_rows_used_for_fit": int(len(hard_train)),
        "hard_negative_validation_rows": int(len(hard_validation)),
        "hard_negative_holdout_rows": int(len(hard_test)),
        "hard_negative_sample_weight": HARD_NEGATIVE_SAMPLE_WEIGHT,
        "hard_negative_domains_excluded_from_benchmark": int(hard_negative_overlap_rows),
        "features": FEATURE_NAMES, "feature_extraction": "URL string only; no network or page access",
        "root_path_canonicalization": "Canonicalized empty paths and '/' as the same model input." if canonicalize_root_slash else "not used",
        "preparation_stats": prep_stats,
    }
    model_comparison = {
        "models": comparisons, "selected_model": selected_name, "selection_rule": selection_rule,
        "split": split_info, "thresholds_considered": list(THRESHOLDS), "selected_threshold": CURRENT_THRESHOLD,
        "final_test_touched_after_selection_only": True, "final_test_results": test_metrics,
        "baseline_on_same_final_test": baseline_comparison,
        "fresh_candidates_on_same_final_test": fresh_candidate_test_metrics,
        "hard_negative_validation_results": hard_validation_metrics,
        "hard_negative_results": hard_metrics, "baseline_hard_negative_results": baseline_hard_metrics,
        "same_partition_model_version_comparison": version_comparisons,
        "canonicalize_root_slash": canonicalize_root_slash,
        "category_evaluation": evaluation,
        "resource_measurements": {"candidate_model_size_bytes": candidate_path.stat().st_size, "process_peak_rss_bytes": int(peak_rss) if peak_rss else None, "memory_note": "Peak process RSS during the whole training run, not a per-model inference measurement."},
    }
    metadata = {
        "model_name": selected_name, "version": candidate_release, "candidate_path": str(candidate_path),
        "training_source": "real", "training_timestamp_utc": now, "training_dataset_sources": source_names,
        "dataset_rows": len(data), "dataset_class_counts": inspection["class_counts"], "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_configuration": {"character_ngrams": "2-5, min_df=2, max_features=100000, sublinear_tf=true", "word_ngrams": "1-2, min_df=2, max_features=60000, used only in word+character model", "numeric_features": FEATURE_NAMES},
        "validation_strategy": split_info, "test_strategy": "Untouched registrable-domain-disjoint partition; selected model and threshold were fixed before final test scoring.",
        "selected_thresholds": selected_thresholds, "calibration": comparisons[selected_name]["calibration"],
        "metrics": test_metrics, "hard_negative_metrics": hard_metrics,
        "hard_negative_validation_metrics": hard_validation_metrics,
        "baseline_on_same_final_test": baseline_comparison,
        "baseline_hard_negative_metrics": baseline_hard_metrics,
        "same_partition_model_version_comparison": version_comparisons,
        "python_version": __import__("platform").python_version(),
        "library_versions": {"scikit_learn": __import__("sklearn").__version__, "pandas": pd.__version__, "numpy": np.__version__, "tldextract": __import__("tldextract").__version__},
        "latency_and_size": comparisons[selected_name], "candidate_status": "not promoted automatically",
    }
    (MODEL_DIR / f"url_model_{version_tag}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (MODEL_DIR / f"url_metrics_{version_tag}.json").write_text(json.dumps(model_comparison, indent=2) + "\n", encoding="utf-8")
    (MODEL_DIR / f"url_dataset_inspection_{version_tag}.json").write_text(json.dumps(inspection, indent=2) + "\n", encoding="utf-8")
    (MODEL_DIR / f"url_model_comparison_{version_tag}.json").write_text(json.dumps(model_comparison, indent=2) + "\n", encoding="utf-8")
    docs = PROJECT_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / f"url_error_analysis_{version_tag}.md").write_text(_error_analysis(test_rows, test_features, hard_rows), encoding="utf-8")
    (docs / f"url_model_improvement_report_{version_tag}.md").write_text(_report(metadata, comparisons, evaluation, inspection, split_info, candidate_path), encoding="utf-8")
    return {"selected_model": selected_name, "threshold": CURRENT_THRESHOLD, "candidate": str(candidate_path), "metrics": test_metrics, "hard_negative": {key: hard_metrics.get(key) for key in ("sample_count", "model_warning_fpr_at_selected_threshold", "confusion_matrix")}}


def _report(metadata, comparisons, evaluation, inspection, split_info, candidate_path) -> str:
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate", "brier_score")
    lines = [
        "# URL model improvement report", "",
        f"Candidate: `{candidate_path.name}` ({metadata['model_name']}, version {metadata['version']}). The existing default model was preserved.", "",
        "## Dataset and split", "",
        f"Rows: {inspection['prepared_rows']}; legitimate {inspection['class_counts']['legitimate']}; phishing {inspection['class_counts']['phishing']}; registrable domains {inspection['unique_registrable_domains']}.", "",
        f"Split: {split_info['method']}. Rows by partition: {split_info['rows']}. Domains do not cross fit, calibration, validation, or test.",
        f"Separate hard-negative partitions: fit {split_info['hard_negative_train_rows']} rows / {split_info['hard_negative_train_domains']} domains; validation {split_info['hard_negative_validation_rows']} / {split_info['hard_negative_validation_domains']} domains; final challenge test {split_info['hard_negative_test_rows']} / {split_info['hard_negative_test_domains']} domains. Hard-negative roots were removed from the broad UCI corpus. Fit examples receive sample weight {HARD_NEGATIVE_SAMPLE_WEIGHT:g}.", "",
        "No timestamped rows were available, so this run has no temporal result.", "",
        "## Model comparison on the same validation domains", "",
        "| Model | F1 at 0.50 | PR-AUC | FPR | Brier | Mean ms | P95 ms | Compressed bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in comparisons.items():
        m = result["validation_metrics_at_0_50"]
        lines.append(f"| {name} | {m['f1']:.4f} | {m['pr_auc']:.4f} | {m['false_positive_rate']:.4%} | {m['brier_score']:.5f} | {result['inference_latency_ms_mean']:.3f} | {result['inference_latency_ms_p95']:.3f} | {result['model_size_bytes_compressed']} |")
    versions = metadata.get("same_partition_model_version_comparison", {})
    lines.extend([
        "", "## v2, v3, and improved-candidate comparison", "",
        "All three artifacts are scored on the same UCI final rows and fresh external legitimate challenge rows. The v4 threshold is selected on validation only; v2/v3 use their previously saved thresholds. Because v2/v3 trained on the underlying UCI corpus, their UCI rows are retrospective and may overlap legacy training. The fresh external challenge roots do not overlap UCI or the prior hard-negative domains. No per-domain exception is applied.", "",
        "| Version | Threshold | Accuracy | Phishing precision | Phishing recall | Phishing F1 | UCI FPR | UCI FNR | Hard-negative FPR | Mean inference ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for version in ("v2", "v3", f"v{metadata['version'].split('.')[0]}"):
        result = versions.get(version, {})
        if not result or "overall_test_metrics" not in result:
            continue
        overall = result["overall_test_metrics"]
        hard = result["hard_negative_test_metrics"]
        latency = result["latency_and_size"]
        lines.append(
            f"| {version} ({result['model_name']}) | {result['threshold_used']:.2f} | {overall['accuracy']:.4f} | {overall['precision']:.4f} | {overall['recall']:.4f} | {overall['f1']:.4f} | {overall['false_positive_rate']:.3%} | {overall['false_negative_rate']:.3%} | {hard.get('model_warning_fpr_at_selected_threshold', 0):.2%} | {latency['inference_latency_ms_mean']:.3f} | {latency['inference_latency_ms_p95']:.3f} |"
        )
    lines.extend(["", "### Final confusion matrices", "", "| Version | UCI TN/FP/FN/TP | Hard-negative TN/FP/FN/TP |", "|---|---|---|"])
    for version in ("v2", "v3", f"v{metadata['version'].split('.')[0]}"):
        result = versions.get(version, {})
        if "overall_test_metrics" in result:
            lines.append(f"| {version} | {result['overall_test_metrics']['confusion_matrix']} | {result['hard_negative_test_metrics'].get('confusion_matrix')} |")
    test = metadata["metrics"]
    baseline = metadata.get("baseline_on_same_final_test", {})
    baseline_test = baseline.get("same_test_at_candidate_selected_threshold", {})
    lines.extend([
        "", "## Selected candidate and untouched test", "",
        f"Selected on validation only: `{metadata['model_name']}` at threshold {metadata['selected_thresholds']['suspicious_probability']:.2f}. Rule: {metadata['selected_thresholds']['selection_rule']}", "",
        "| Metric | Test result |", "|---|---:|",
    ])
    lines.extend(f"| {name} | {test[name]:.4%} |" if name in ("false_positive_rate", "false_negative_rate") else f"| {name} | {test[name]:.4f} |" for name in metric_names if test.get(name) is not None)
    lines.extend([
        f"| Support | {test['support']} |", f"| Confusion matrix TN/FP/FN/TP | {test['confusion_matrix']} |", "",
        "### Existing model on the same final test", "",
        f"At the candidate threshold {metadata['selected_thresholds']['suspicious_probability']:.2f}, existing model F1={baseline_test.get('f1'):.4f}, FPR={baseline_test.get('false_positive_rate'):.4%}, FNR={baseline_test.get('false_negative_rate'):.4%}, confusion={baseline_test.get('confusion_matrix')}. This same-test comparison was computed after candidate selection and did not affect it.", "",
        "## Separate hard-negative results", "",
        f"Fresh external official legitimate challenge URLs on unseen roots: {evaluation['hard_negative_legitimate_urls'].get('sample_count', 0)}. Candidate false-positive rate at its selected validation threshold: {evaluation['hard_negative_legitimate_urls'].get('model_warning_fpr_at_selected_threshold', 0):.2%}.",
        f"Existing v2 model false-positive rate on the same examples at its saved threshold: {metadata.get('baseline_hard_negative_metrics', {}).get('model_warning_fpr_at_selected_threshold'):.2%}.",
        f"Hard-negative validation FPR at the selected threshold: {metadata.get('hard_negative_validation_metrics', {}).get('model_warning_fpr_at_selected_threshold'):.2%}. Model and threshold selection prioritized hard-negative validation FPR under the broad-validation FPR cap; neither final test set informed selection.", "",
        "| Hard-negative category | Samples | FPR | Confusion matrix TN/FP/FN/TP |",
        "|---|---:|---:|---:|",
    ])
    categories = evaluation["hard_negative_legitimate_urls"].get("categories", {})
    for category, result in categories.items():
        fpr = result.get('false_positive_rate')
        lines.append(f"| {category} | {result.get('sample_count', 0)} | {fpr:.2%} | {result.get('confusion_matrix')} |")
    lines.extend([
        "", "## Calibration", "",
        f"Selected method: {metadata['calibration']['method']}. Sigmoid validation Brier: {metadata['calibration'].get('sigmoid_validation_brier')}; isotonic validation Brier: {metadata['calibration'].get('isotonic_validation_brier')}; calibration partition is separate from validation. Final-test Brier: {test['brier_score']:.5f}.", "",
        "## Limitations and recommendations", "",
        "The underlying UCI dataset is a static 2024 snapshot and has no per-URL timestamps. This is domain-generalization evaluation, not a measure of current live phishing coverage. The hard-negative challenge is mined from the UCI legitimate class and is not a population sample; independently verified recent benign URLs would provide a stronger challenge. The candidate is saved separately and is not promoted automatically. For production, add a dated, authorized phishing feed and a separately sourced dated benign URL stream, preserve label snapshots, retrain on older dates, and evaluate the newest date window without threshold tuning on it.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare URL-only phishing classifiers with registrable-domain-safe partitions; keep candidate artifacts separate.")
    parser.add_argument("--source", choices=("prepared", "csv", "uci", "uci-csv"), default="prepared")
    parser.add_argument("--csv", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_PREPARED_HARD_NEGATIVES)
    parser.add_argument("--candidate-version", type=int, choices=(3, 4, 5, 6), default=3)
    args = parser.parse_args()
    if args.source == "csv" and args.csv == DEFAULT_PREPARED and not args.csv.exists():
        parser.error("Run backend/prepare_url_dataset.py first or pass --csv")
    result = train(args.csv, args.hard_negatives, candidate_version=args.candidate_version)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
