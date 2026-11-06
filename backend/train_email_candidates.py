"""Train and compare source-held-out real email candidates without replacing v2."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.frozen import FrozenEstimator

from email_features import clean_email_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend/models"
SOURCE_CSV = ROOT / "data/raw/phishing_emails.csv"
CHALLENGE_CSV = ROOT / "data/email_hard_negative_challenge.csv"
SEED = 4207
THRESHOLDS = tuple(round(float(t), 3) for t in np.arange(0.10, 0.991, 0.01))
DISPLAY_THRESHOLDS = (0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99)


def load_data() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(SOURCE_CSV, low_memory=False)
    cols = {str(c).lower(): c for c in raw.columns}
    label_map = {"1": 1, "phishing": 1, "phish": 1, "malicious": 1, "spam": 1,
        "yes": 1, "0": 0, "legitimate": 0, "safe": 0, "ham": 0, "benign": 0, "no": 0}
    label_col = cols.get("label") or cols.get("target") or cols.get("class")
    subject_col = cols.get("subject")
    body_col = cols.get("body") or cols.get("text") or cols.get("message")
    source_col = cols.get("source")
    if not label_col or not (subject_col or body_col):
        raise ValueError("Email CSV requires labels and subject/body text")
    labels = raw[label_col].map(lambda x: label_map.get(str(x).strip().lower()))
    subject = raw[subject_col].fillna("").astype(str) if subject_col else pd.Series("", index=raw.index)
    body = raw[body_col].fillna("").astype(str) if body_col else pd.Series("", index=raw.index)
    text = [clean_email_text(s, b) for s, b in zip(subject, body)]
    source = raw[source_col].fillna("unspecified").astype(str) if source_col else pd.Series("unspecified", index=raw.index)
    frame = pd.DataFrame({"text": text, "subject": subject, "label": labels,
        "source": source, "row_id": np.arange(len(raw))}).dropna(subset=["label"])
    frame["label"] = frame.label.astype(int)
    frame = frame[frame.text.ne("")].drop_duplicates("text").reset_index(drop=True)
    frame["subject_group"] = [subject_group(s, t, i) for s, t, i in zip(frame.subject, frame.text, frame.row_id)]
    if frame.label.nunique() != 2:
        raise ValueError("Email corpus must include both labels")
    snapshot = hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest()
    inspection = {
        "dataset": str(SOURCE_CSV.relative_to(ROOT)), "sha256": snapshot,
        "raw_rows": int(len(raw)), "clean_unique_rows": int(len(frame)),
        "invalid_or_missing_labels": int(labels.isna().sum()),
        "missing_values": {str(k): int(raw[v].isna().sum()) for k, v in cols.items() if k in {"subject", "body", "label", "source"}},
        "exact_cleaned_duplicates_removed": int(len(raw) - len(frame)),
        "conflicting_cleaned_text_labels": int(frame.groupby("text").label.nunique().gt(1).sum()),
        "class_counts": {"benign": int((frame.label == 0).sum()), "phishing": int((frame.label == 1).sum())},
        "source_counts": {str(k): int(v) for k, v in frame.source.value_counts().items()},
        "source_label_counts": {str(src): {"benign": int((g.label == 0).sum()), "phishing": int((g.label == 1).sum())} for src, g in frame.groupby("source")},
        "distinct_normalized_subjects": int(frame.subject_group.nunique()),
        "label_quality_note": "Labels are inherited from the public multi-source corpus; individual labels have not been independently adjudicated.",
    }
    return frame, inspection


def subject_group(subject: str, text: str, row_id: int) -> str:
    normalized = clean_email_text(subject, "")
    if normalized:
        return "subject:" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return "empty-subject:" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_data(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    source = frame.source.to_numpy()
    labels = frame.label.to_numpy(dtype=int)
    test = np.flatnonzero(np.char.lower(source.astype(str)) == "trec5")
    train_pool = np.flatnonzero(np.char.lower(source.astype(str)) != "trec5")
    if not len(test) or len(np.unique(labels[test])) != 2:
        raise ValueError("Expected the TREC-05 final holdout with both classes")
    test_subjects = set(frame.subject_group.iloc[test])
    overlapping = frame.subject_group.iloc[train_pool].isin(test_subjects).to_numpy()
    retained = train_pool[~overlapping]
    excluded = train_pool[overlapping]
    groups = frame.subject_group.to_numpy()[retained]

    # Fit/calibration/validation groups have disjoint normalized subjects.
    # TREC-05 remains sealed while split choice and threshold tuning happen.
    for seed in range(SEED, SEED + 60):
        try:
            fit_cal_local, val_local = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(retained, labels[retained], groups))
            fit_cal_global = retained[fit_cal_local]
            fit_rel, cal_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed + 100).split(fit_cal_global, labels[fit_cal_global], frame.subject_group.to_numpy()[fit_cal_global]))
        except StopIteration:
            continue
        partitions = {"fit": fit_cal_global[fit_rel], "calibration": fit_cal_global[cal_rel], "validation": retained[val_local], "test": test, "subject_overlap_excluded": excluded}
        if all(len(np.unique(labels[partitions[k]])) == 2 for k in ("fit", "calibration", "validation")):
            subject_sets = {k: set(frame.subject_group.iloc[v]) for k, v in partitions.items() if k not in {"test", "subject_overlap_excluded"}}
            if all(not subject_sets[a] & subject_sets[b] for a in subject_sets for b in subject_sets if a < b):
                return partitions, {
                    "method": "TREC-05 source-disjoint final test; development fit/calibration/validation split by normalized exact subject groups; development subjects also appearing in TREC-05 are purged",
                    "seed": seed,
                    "rows": {k: int(len(v)) for k, v in partitions.items()},
                    "sources": {k: sorted(set(source[v])) for k, v in partitions.items() if k != "subject_overlap_excluded"},
                    "subject_overlap_counts_fit_calibration_validation": {f"{a}_{b}": len(subject_sets[a] & subject_sets[b]) for a in subject_sets for b in subject_sets if a < b},
                    "test_subject_overlap_with_fit_calibration_validation": 0,
                    "subject_overlap_removed_from_development": int(len(excluded)),
                    "not_measured": "Verified campaign identity and all semantic near-duplicate clusters are not available in source fields; this split prevents exact subject reuse but cannot guarantee campaign-disjoint TREC-05 evaluation.",
                }
    raise ValueError("Could not create subject-group-disjoint fit/calibration/validation partitions with both labels")


def models() -> dict[str, Pipeline]:
    lr = lambda C, weight: LogisticRegression(C=C, max_iter=1800, class_weight=weight, random_state=SEED)
    word_current = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True)
    word_improved = lambda: TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=0.90, max_features=80_000, sublinear_tf=True)
    def word_char():
        word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.90, max_features=60_000, sublinear_tf=True)
        char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=80_000, sublinear_tf=True)
        return FeatureUnion([("word", word), ("char", char)], transformer_weights={"word": 1.0, "char": 0.5}, n_jobs=1)
    return {
        "word_1_2_balanced_baseline": Pipeline([("tfidf", word_current()), ("clf", lr(1.0, "balanced"))]),
        "word_1_3_balanced": Pipeline([("tfidf", word_improved()), ("clf", lr(0.7, "balanced"))]),
        "word_1_3_unbalanced": Pipeline([("tfidf", word_improved()), ("clf", lr(0.7, None))]),
        "word_char_balanced": Pipeline([("tfidf", word_char()), ("clf", lr(1.0, "balanced"))]),
    }


def probability(model, texts) -> np.ndarray:
    classes = list(model.classes_)
    return np.asarray(model.predict_proba(texts)[:, classes.index(1)], dtype=float)


def metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if tn + fp else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"benign": int((y == 0).sum()), "phishing": int((y == 1).sum())},
    }


def choose_threshold(y, p) -> tuple[float, list[dict]]:
    rows = [metrics(y, p, t) for t in THRESHOLDS]
    # Validation-only operating rule: preserve at least 95% recall, then
    # minimize FPR. Tie-break on precision and F1.
    eligible = [r for r in rows if r["recall"] >= 0.95]
    selected = min(eligible, key=lambda r: (r["false_positive_rate"], -r["precision"], -r["f1"], -r["recall"])) if eligible else max(rows, key=lambda r: (r["f1"], r["precision"]))
    return selected["threshold"], rows


def load_challenge() -> pd.DataFrame:
    data = pd.read_csv(CHALLENGE_CSV)
    data["text"] = [clean_email_text(s, b) for s, b in zip(data.subject, data.body)]
    return data


def main() -> None:
    frame, inspection = load_data()
    partitions, split_info = split_data(frame)
    y = frame.label.to_numpy(dtype=int)
    texts = frame.text.to_numpy()
    challenge = load_challenge()
    model_metrics = {}
    fitted = {}

    for name, base in models().items():
        started = time.perf_counter()
        fit_idx, cal_idx, val_idx = (partitions[k] for k in ("fit", "calibration", "validation"))
        base.fit(texts[fit_idx], y[fit_idx])
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrated.fit(texts[cal_idx], y[cal_idx])
        val_p = probability(calibrated, texts[val_idx])
        threshold, sweep = choose_threshold(y[val_idx], val_p)
        selected_val = metrics(y[val_idx], val_p, threshold)
        displayed = {str(t): metrics(y[val_idx], val_p, t) for t in DISPLAY_THRESHOLDS}
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            size_path = Path(tmp.name)
        try:
            joblib.dump(calibrated, size_path, compress=3)
            size = int(size_path.stat().st_size)
        finally:
            size_path.unlink(missing_ok=True)
        samples = texts[val_idx[: min(200, len(val_idx))]]
        durations = []
        for text in samples:
            t0 = time.perf_counter(); probability(calibrated, [text]); durations.append((time.perf_counter() - t0) * 1000)
        model_metrics[name] = {
            "validation_at_selected_threshold": selected_val,
            "validation_threshold_sweep": sweep,
            "validation_display_thresholds": displayed,
            "threshold_selected_on_validation": threshold,
            "calibration": {"method": "sigmoid", "validation_brier": float(brier_score_loss(y[val_idx], val_p))},
            "model_size_bytes_compressed": size,
            "inference_latency_ms_mean": float(np.mean(durations)),
            "inference_latency_ms_p95": float(np.percentile(durations, 95)),
            "fit_calibrate_validate_seconds": float(time.perf_counter() - started),
        }
        fitted[name] = calibrated
        print(f"Compared {name}", flush=True)

    eligible = [n for n, d in model_metrics.items() if d["validation_at_selected_threshold"]["recall"] >= 0.95]
    selected_name = min(eligible, key=lambda n: (
        model_metrics[n]["validation_at_selected_threshold"]["false_positive_rate"],
        -model_metrics[n]["validation_at_selected_threshold"]["precision"],
        -model_metrics[n]["validation_at_selected_threshold"]["f1"],
        model_metrics[n]["model_size_bytes_compressed"],
    )) if eligible else max(model_metrics, key=lambda n: model_metrics[n]["validation_at_selected_threshold"]["f1"])
    selected = fitted[selected_name]
    selected_threshold = model_metrics[selected_name]["threshold_selected_on_validation"]

    # Thresholds and model choice are now frozen. Only now touch TREC-05.
    for name, model in fitted.items():
        test_p = probability(model, texts[partitions["test"]])
        model_metrics[name]["heldout_test_at_validation_threshold"] = metrics(
            y[partitions["test"]], test_p,
            model_metrics[name]["threshold_selected_on_validation"],
        )

    challenge_metrics = {}
    for name, model in fitted.items():
        p = probability(model, challenge.text)
        threshold = model_metrics[name]["threshold_selected_on_validation"]
        challenge_metrics[name] = {
            "metrics": metrics(challenge.expected_label, p, threshold),
            "predictions": [{"category": r.category, "expected_label": int(r.expected_label), "probability": float(prob), "prediction": int(prob >= threshold)} for r, prob in zip(challenge.itertuples(), p)],
            "note": "Manually authored challenge kept out of fit, calibration, validation, and model selection.",
        }

    # Preserve all existing production email artifacts. Save only an opt-in
    # candidate, which is promoted by a separate reviewed integration step.
    release = "3.0.0-candidate"
    artifact = {
        "model": selected, "risk_classes": {"malicious": 1},
        "training_source": "real", "model_name": selected_name,
        "model_version": release, "probability_calibrated": True,
        "calibration_method": "sigmoid",
        "decision_thresholds": {"phishing_probability": selected_threshold},
        "candidate_status": "not promoted; current email_model.joblib remains unchanged",
    }
    candidate_path = MODEL_DIR / "email_model_v3_candidate.joblib"
    joblib.dump(artifact, candidate_path, compress=3)
    metadata = {
        "model_name": selected_name, "version": release, "candidate_path": str(candidate_path),
        "training_source": "real", "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_inspection": inspection, "split": split_info,
        "candidate_selection_rule": "Among models with at least 95% validation recall, minimize validation FPR; ties prefer precision, F1, then model size. TREC-05 and the hard-negative challenge are excluded from selection.",
        "selected_model": selected_name, "selected_threshold": selected_threshold,
        "models": model_metrics, "challenge": challenge_metrics,
        "heldout_final_test_touched_after_selection_only": True,
    }
    out_json = MODEL_DIR / "email_model_v3_candidate_metrics.json"
    out_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path = ROOT / "docs/email_model_improvement_report_v3.md"
    lines = ["# Email model candidate comparison v3", "", "This is an opt-in candidate report. The production `email_model.joblib` and v2 metadata were preserved.", "",
        f"Dataset SHA-256: `{inspection['sha256']}`. Final test: TREC-05 ({len(partitions['test']):,} rows). Fit/calibration/validation use TREC-06/07 after exact subject overlap with TREC-05 is excluded. Model choice and threshold selection used validation only.", "",
        f"Selected candidate: **{selected_name}**, probability threshold **{selected_threshold:.3f}**.", "",
        "| Model | Threshold | Validation precision | Validation recall | Validation FPR | Validation F1 | Test accuracy | Test precision | Test recall | Test FPR | Test FNR | ROC-AUC | Brier | Latency ms | Size MB |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, d in model_metrics.items():
        v, t = d["validation_at_selected_threshold"], d["heldout_test_at_validation_threshold"]
        lines.append(f"| {name} | {d['threshold_selected_on_validation']:.3f} | {v['precision']:.4%} | {v['recall']:.4%} | {v['false_positive_rate']:.4%} | {v['f1']:.4%} | {t['accuracy']:.4%} | {t['precision']:.4%} | {t['recall']:.4%} | {t['false_positive_rate']:.4%} | {t['false_negative_rate']:.4%} | {t['roc_auc']:.5f} | {t['brier_score']:.5f} | {d['inference_latency_ms_mean']:.3f} | {d['model_size_bytes_compressed']/1_000_000:.2f} |")
    lines += ["", f"Current v2 held-out benchmark from `email_metadata.json` (historical snapshot): accuracy 85.34%, precision 75.17%, recall 96.23%, FPR 22.29%, FNR 3.77%, ROC-AUC 0.96742, Brier 0.10721 at probability 0.50. The current workspace CSV differs by 14 cleaned rows from that saved training snapshot; this comparison is indicative, while comparisons among these candidates share the current snapshot/split.", "",
        "## Dataset and split audit", "", f"- Clean unique rows: {inspection['clean_unique_rows']:,}; benign {inspection['class_counts']['benign']:,}, phishing {inspection['class_counts']['phishing']:,}.", f"- Exact cleaned duplicates removed before split: {inspection['exact_cleaned_duplicates_removed']:,}; conflicting cleaned-text labels: {inspection['conflicting_cleaned_text_labels']}.", f"- Rows removed from TREC-06/07 because normalized subjects also occur in TREC-05: {split_info['subject_overlap_removed_from_development']:,}.", f"- Subject groups are disjoint across fit/calibration/validation and from TREC-05. Campaign IDs are absent; near-duplicate/template isolation cannot be guaranteed. The prior read-only audit found 135 TREC-05 LSH near-duplicate candidates among 8,123 candidate-bearing messages at estimated signature overlap >=0.75; these are not confirmed semantic duplicates.", "",
        "## Challenge set", "", "The manually authored 18-message challenge is separate from the public corpus and all model selection. It includes account/storage/security notifications, transactions, newsletters, meetings, and phishing language. Results are descriptive for this small set and are not used as population metrics.", "",
        "## Limits", "", "The TREC-05 source holdout is from a different source collection, not a time-based evaluation. Exact subject overlap is removed from training; verified campaign and semantic-template overlap is not available. Public labels may be wrong. Email-only metrics exclude the API's separate URL score; HTML parsing and URL extraction remain as implemented. The candidate is not automatically promoted.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "threshold": selected_threshold,
        "candidate_path": str(candidate_path), "metrics_path": str(out_json),
        "heldout_test": model_metrics[selected_name]["heldout_test_at_validation_threshold"],
        "challenge": challenge_metrics[selected_name]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
