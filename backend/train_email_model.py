from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from email_features import clean_email_text

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SEED = 42


def load_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, low_memory=False)
    cols = {str(column).lower(): column for column in frame.columns}
    label_col = cols.get("label") or cols.get("target") or cols.get("class")
    subject_col = cols.get("subject")
    body_col = cols.get("body") or cols.get("text") or cols.get("message") or cols.get("text_combined")
    source_col = cols.get("source")
    if not label_col or not (body_col or subject_col):
        raise ValueError("CSV needs a label and at least one of subject/body/text/message columns")
    label_map = {"1": 1, "phishing": 1, "phish": 1, "malicious": 1, "spam": 1, "yes": 1,
                 "0": 0, "legitimate": 0, "safe": 0, "ham": 0, "benign": 0, "no": 0}
    subject = frame[subject_col].fillna("").astype(str) if subject_col else pd.Series([""] * len(frame), index=frame.index)
    body = frame[body_col].fillna("").astype(str) if body_col else pd.Series([""] * len(frame), index=frame.index)
    labels = frame[label_col].map(lambda value: label_map.get(str(value).strip().lower()))
    data = pd.DataFrame({
        "text": [clean_email_text(s, b) for s, b in zip(subject, body)],
        "label": labels,
        "source": frame[source_col].fillna("unspecified").astype(str) if source_col else "unspecified",
    }).dropna(subset=["label"])
    data["label"] = data.label.astype(int)
    data = data[data.text.ne("")].drop_duplicates("text").reset_index(drop=True)
    if data.label.nunique() != 2:
        raise ValueError("Dataset must include both safe and phishing labels")
    inspection = {
        "rows_after_cleaning": len(data),
        "duplicate_texts_removed": int(frame.shape[0] - len(data)),
        "label_mapping": "0=benign, 1=phishing",
        "label_counts": {"benign": int((data.label == 0).sum()), "phishing": int((data.label == 1).sum())},
        "source_counts": {str(k): int(v) for k, v in data.source.value_counts().items()},
        "source_label_counts": {str(source): {str(int(label)): int(count) for label, count in group.label.value_counts().items()} for source, group in data.groupby("source")},
    }
    return data, inspection


def _pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1800, C=1.0, class_weight="balanced", random_state=SEED)),
    ])


def _group_split(text, y, groups, fraction: float, seed: int):
    # GroupShuffleSplit keeps every email collection wholly within one partition.
    for train_idx, test_idx in GroupShuffleSplit(n_splits=30, test_size=fraction, random_state=seed).split(text, y, groups):
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
            return train_idx, test_idx
    raise ValueError("Cannot make source-disjoint splits containing both labels; use a dataset with both classes in multiple sources")


def _calibration_split(y, fraction: float = 0.15, seed: int = SEED):
    """Reserve a stratified slice from the training sources for probability calibration.

    The final test collection remains source-disjoint. Splitting calibration rows
    within the remaining sources lets the classifier learn from every available
    collection instead of sacrificing an entire source to calibration.
    """
    indices = np.arange(len(y))
    return train_test_split(indices, test_size=fraction, random_state=seed, stratify=y)


def _probability(model, texts) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(texts)[:, classes.index(1)]


def _metrics(y_true, probability) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(probability) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"benign": int((y_true == 0).sum()), "phishing": int((y_true == 1).sum())},
    }


def demo_frame() -> pd.DataFrame:
    safe = ["Weekly project update the meeting starts tomorrow", "Your package has shipped and will arrive Friday", "Class timetable is posted on the student portal", "Thanks for attending the engineering review", "Your monthly statement is ready in the banking app", "The support team received your request", "Your subscription renews next month", "Please review the attached project notes", "Your interview is confirmed for Monday", "We received your payment thank you"]
    phishing = ["URGENT verify your account password immediately or it will be closed", "Your bank account is locked click to restore access now", "Confirm your credentials to avoid suspension today", "Payment failed enter your card number and OTP immediately", "CEO urgent request buy gift cards and send the codes", "Your mailbox will be disabled unless you verify your identity", "Wallet verification required submit your recovery phrase", "Click this link and confirm your bank details now", "Security alert sign in to stop account deletion", "Invoice overdue pay immediately through the verification page"]
    return pd.DataFrame({"text": [clean_email_text("", x) for x in safe + phishing], "label": [0] * len(safe) + [1] * len(phishing), "source": ["DEMO"] * 20})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression on real labeled email data, or explicitly opt into the DEMO fixture.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=Path, help="CSV with label and subject/body or text columns")
    group.add_argument("--demo", action="store_true", help="Train on tiny synthetic examples for software demos only")
    args = parser.parse_args()
    if args.demo:
        raise SystemExit("DEMO artifacts are created by backend/train_all.py; this script never writes synthetic data to real model paths.")
    frame, inspection = load_csv(args.csv)
    texts = frame.text.to_numpy()
    labels = frame.label.to_numpy()
    groups = frame.source.to_numpy()
    train_pool, test_idx = _group_split(texts, labels, groups, 0.20, SEED)
    fit_rel, val_rel = _calibration_split(labels[train_pool], seed=SEED + 1)
    fit_idx, val_idx = train_pool[fit_rel], train_pool[val_rel]
    if set(groups[fit_idx]) & set(groups[test_idx]) or set(groups[val_idx]) & set(groups[test_idx]):
        raise RuntimeError("Email test-source leakage detected")
    if any(len(np.unique(labels[idx])) != 2 for idx in (fit_idx, val_idx, test_idx)):
        raise ValueError("Each source-disjoint split must contain both classes")
    base = _pipeline()
    base.fit(texts[fit_idx], labels[fit_idx])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(texts[val_idx], labels[val_idx])
    probability = _probability(calibrated, texts[test_idx])
    metrics = _metrics(labels[test_idx], probability)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {"model": calibrated, "risk_classes": {"malicious": 1}, "training_source": "real", "model_name": "TF-IDF + calibrated Logistic Regression", "model_version": "2.1.0", "probability_calibrated": True, "calibration_method": "sigmoid", "decision_thresholds": {"safe_max": 29, "suspicious_max": 69}}
    joblib.dump(artifact, MODEL_DIR / "email_model.joblib")
    joblib.dump(base.named_steps["tfidf"], MODEL_DIR / "email_vectorizer.joblib")
    test_sources = sorted(set(groups[test_idx]))
    metadata = {
        "model_name": artifact["model_name"], "version": artifact["model_version"],
        "dataset": str(args.csv), "training_source": "real", "training_date": datetime.now(timezone.utc).isoformat(),
        "feature_count": int(len(base.named_steps["tfidf"].vocabulary_)), "rows": int(len(frame)),
        "split_method": "Source-disjoint final test; stratified calibration holdout drawn from the training sources",
        "fit_rows": int(len(fit_idx)), "validation_rows": int(len(val_idx)), "test_rows": int(len(test_idx)),
        "fit_sources": sorted(set(groups[fit_idx])), "validation_sources": sorted(set(groups[val_idx])), "test_sources": test_sources,
        "metrics": metrics, "metrics_are_population_claims": False,
        "dataset_inspection": inspection,
    }
    (MODEL_DIR / "email_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODEL_DIR / "email_training_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
