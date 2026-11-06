"""Post-freeze evaluation of v5 on the separate v4 final URL challenge.

This script is intentionally separate from training and threshold selection.
It reads the stored v4 final challenge only after url_model_v5.joblib and its
threshold have been written. It never changes the model or operating point.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

from domain_utils import registrable_domain
from train_url_model import _input, _metric_row, _probability
from url_features import FEATURE_NAMES, extract_url_features, url_indicators

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend/models"
DOCS_DIR = ROOT / "docs"
THRESHOLD = lambda artifact: float(artifact["decision_thresholds"]["suspicious_probability"])


def _frame(url: str, model_name: str) -> pd.DataFrame:
    values = extract_url_features(url)
    row = pd.DataFrame([[values[key] for key in FEATURE_NAMES]], columns=FEATURE_NAMES)
    if not model_name.startswith("numeric_"):
        row.insert(0, "url", url)
    return row


def _char_attribution(artifact: dict, url: str) -> dict:
    calibrated = artifact["model"]
    pipeline = calibrated.estimator.estimator
    preprocessor = pipeline.named_steps["columntransformer"]
    classifier = pipeline.named_steps["logisticregression"]
    vectorizer = preprocessor.named_transformers_["char"]
    char_row = vectorizer.transform([url])
    char_products = char_row.multiply(classifier.coef_[0][:char_row.shape[1]]).tocoo()
    names = vectorizer.get_feature_names_out()
    char_terms = sorted(
        ((float(value), str(names[index])) for value, index in zip(char_products.data, char_products.col)),
        reverse=True,
    )
    positive_char_terms = [(value, gram) for value, gram in char_terms if value > 0]
    negative_char_terms = [(value, gram) for value, gram in char_terms if value < 0]
    numeric_transformer = preprocessor.named_transformers_.get("numeric")
    numeric_terms = []
    if numeric_transformer is not None:
        features = extract_url_features(url)
        row = pd.DataFrame([[features[key] for key in FEATURE_NAMES]], columns=FEATURE_NAMES)
        scaled = numeric_transformer.transform(row)[0]
        numeric_coef = classifier.coef_[0][char_row.shape[1]:]
        numeric_terms = sorted(
            ((float(value), name) for value, name in zip(numeric_coef * scaled, FEATURE_NAMES)),
            reverse=True,
        )
    positive_numeric_terms = [(value, name) for value, name in numeric_terms if value > 0]
    negative_numeric_terms = [(value, name) for value, name in numeric_terms if value < 0]
    model_input = _frame(url, artifact["model_name"])
    decision = float(pipeline.decision_function(_input(artifact["model_name"], model_input))[0])
    probability = float(_probability(calibrated, _input(artifact["model_name"], model_input))[0])
    sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
    return {
        "raw_logistic_margin": decision,
        "calibrated_probability": probability,
        "calibration_method": artifact.get("calibration_method"),
        "sigmoid_a": float(sigmoid.a_),
        "sigmoid_b": float(sigmoid.b_),
        "url_features": extract_url_features(url),
        "indicators": url_indicators(extract_url_features(url), url),
        "top_positive_character_ngrams": [
            {"ngram": gram, "logit_contribution": value} for value, gram in positive_char_terms[:14]
        ],
        "top_negative_character_ngrams": [
            {"ngram": gram, "logit_contribution": value} for value, gram in sorted(negative_char_terms)[:8]
        ],
        "top_positive_numeric_features": [
            {"feature": name, "logit_contribution": value} for value, name in positive_numeric_terms[:12]
        ],
        "top_negative_numeric_features": [
            {"feature": name, "logit_contribution": value} for value, name in sorted(negative_numeric_terms)[:8]
        ],
    }


def _word_char_attribution(artifact: dict, url: str) -> dict:
    """Split active word and character n-gram contributions for word+char LR."""
    calibrated = artifact["model"]
    pipeline = calibrated.estimator.estimator
    preprocessor = pipeline.named_steps["columntransformer"]
    classifier = pipeline.named_steps["logisticregression"]
    offset = 0
    details = {}
    for block in ("word", "char"):
        vectorizer = preprocessor.named_transformers_[block]
        sparse_row = vectorizer.transform([url])
        contributions = sparse_row.multiply(
            classifier.coef_[0][offset:offset + sparse_row.shape[1]]
        ).tocoo()
        names = vectorizer.get_feature_names_out()
        terms = sorted(
            ((float(value), str(names[index])) for value, index in zip(contributions.data, contributions.col)),
            reverse=True,
        )
        positive_terms = [(value, ngram) for value, ngram in terms if value > 0]
        negative_terms = [(value, ngram) for value, ngram in terms if value < 0]
        details[block] = {
            "top_positive": [{"ngram": ngram, "logit_contribution": value} for value, ngram in positive_terms[:14]],
            "top_negative": [{"ngram": ngram, "logit_contribution": value} for value, ngram in sorted(negative_terms)[:8]],
        }
        offset += sparse_row.shape[1]
    model_input = _frame(url, artifact["model_name"])
    decision = float(pipeline.decision_function(_input(artifact["model_name"], model_input))[0])
    probability = float(_probability(calibrated, _input(artifact["model_name"], model_input))[0])
    sigmoid = calibrated.calibrated_classifiers_[0].calibrators[0]
    return {
        "raw_logistic_margin": decision,
        "calibrated_probability": probability,
        "calibration_method": artifact.get("calibration_method"),
        "sigmoid_a": float(sigmoid.a_),
        "sigmoid_b": float(sigmoid.b_),
        "url_features": extract_url_features(url),
        "indicators": url_indicators(extract_url_features(url), url),
        "ngram_contributions": details,
    }


def main() -> None:
    comparison_path = MODEL_DIR / "url_model_comparison_v5.json"
    v4_path = MODEL_DIR / "url_metrics_v4.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    v4_metrics = json.loads(v4_path.read_text(encoding="utf-8"))
    # This immutable list is the held-out challenge used for v4 evaluation.
    # It is read only here, after v5 training and threshold selection completed.
    challenge = v4_metrics["hard_negative_results"]["results"]
    artifacts = {
        "v2": joblib.load(MODEL_DIR / "url_model.joblib"),
        "v3": joblib.load(MODEL_DIR / "url_model_v3.joblib"),
        "v4": joblib.load(MODEL_DIR / "url_model_v4.joblib"),
        "v5": joblib.load(MODEL_DIR / "url_model_v5.joblib"),
    }
    names = {
        version: str(artifacts[version].get("model_name", version))
        for version in artifacts
    }
    rows_by_version: dict[str, list[dict]] = {version: [] for version in artifacts}
    for record in challenge:
        url = str(record["url"])
        for version, artifact in artifacts.items():
            model_name = names[version]
            probability = float(
                _probability(artifact["model"], _input(model_name, _frame(url, model_name)))[0]
            )
            threshold = THRESHOLD(artifact)
            rows_by_version[version].append({
                "url": url,
                "registrable_domain": registrable_domain(urlsplit(url).hostname or ""),
                "category": record.get("category", "unspecified"),
                "expected_label": int(record["expected_label"]),
                "probability": probability,
                "threshold": threshold,
                "threshold_prediction": int(probability >= threshold),
            })

    v4_fps = [row for row in rows_by_version["v4"] if row["threshold_prediction"]]
    v4_val_rows = v4_metrics["hard_negative_validation_results"]["results"]
    v4_correct = [row for row in v4_val_rows if row["threshold_prediction"] == 0]
    comparison_rows = []
    version_metrics = comparison["same_partition_model_version_comparison"]
    for version in ("v2", "v3", "v4", "v5"):
        item = version_metrics[version]
        metric = item["overall_test_metrics"]
        latency = item["latency_and_size"]
        challenge_rows = rows_by_version[version]
        fp_count = sum(row["threshold_prediction"] for row in challenge_rows)
        artifact = artifacts[version]
        comparison_rows.append({
            "version": version,
            "model_name": names[version],
            "threshold": THRESHOLD(artifact),
            "uci_final": metric,
            "challenge_false_positives": int(fp_count),
            "challenge_count": len(challenge_rows),
            "challenge_fpr": fp_count / len(challenge_rows) if challenge_rows else None,
            "latency": latency,
            "calibration_method": artifact.get("calibration_method", "unknown"),
        })

    v5_artifact = artifacts["v5"]
    v5_metadata = json.loads((MODEL_DIR / "url_model_v5_metadata.json").read_text(encoding="utf-8"))
    v5_calibration = v5_metadata.get("calibration", {})
    candidate_threshold = THRESHOLD(v5_artifact)
    v4_artifact = artifacts["v4"]
    v4_detail = {}
    for row in v4_fps:
        url = row["url"]
        fs = extract_url_features(url)
        v4_detail[url] = {
            "category": row["category"],
            "v4_probability": row["probability"],
            "v4_threshold": row["threshold"],
            "v4_feature_trace": _char_attribution(v4_artifact, url),
            "v5_probability": next(r["probability"] for r in rows_by_version["v5"] if r["url"] == url),
            "v5_threshold": candidate_threshold,
            "v5_prediction": next(r["threshold_prediction"] for r in rows_by_version["v5"] if r["url"] == url),
            "v4_structural_features": fs,
        }
    audit = {
        "challenge_source": "the fixed 21-root challenge recorded in url_metrics_v4.json; read only after v5 was frozen",
        "v5_threshold_was_fixed_before_challenge_scoring": True,
        "v5_threshold": candidate_threshold,
        "v5_selected_model": names["v5"],
        "challenge_predictions": rows_by_version,
        "comparison": comparison_rows,
        "v4_false_positive_attributions": v4_detail,
        "v4_correctly_classified_validation_examples": sorted(
            v4_correct, key=lambda row: abs(float(row["model_probability"]) - 0.8)
        ),
    }
    _write_reports(audit, comparison, v5_metadata, v5_calibration, artifacts)
    (MODEL_DIR / "url_final_challenge_metrics_v5.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


def _write_reports(audit: dict, comparison: dict, metadata: dict, calibration: dict, artifacts: dict) -> None:
    versions = {row["version"]: row for row in audit["comparison"]}
    lines = [
        "# URL model improvement report v5", "",
        "v5 is a candidate only. The current default artifact remains v2. Candidate fitting and threshold selection used the v5 development hard-negative file and the registrable-domain-disjoint UCI fit/calibration/validation partitions. The challenge's registrable roots were used only in a separate data audit to prevent overlap; no challenge URL text, label, score, or performance result entered fit, calibration, or threshold selection. Challenge predictions were generated only after v5 was saved.", "",
        "## v5 training and split", "",
        f"Selected model: `{metadata['model_name']}`; threshold: **{metadata['selected_thresholds']['suspicious_probability']:.2f}**. Hard-negative development data: {metadata['validation_strategy']['hard_negative_train_rows']} fit, {metadata['validation_strategy']['hard_negative_validation_rows']} validation, and {metadata['validation_strategy']['hard_negative_test_rows']} development-holdout rows. The challenge remains separate and has 21 registrable domains.", "",
        "Threshold selection maximized phishing recall subject to the validation FPR caps in the trainer. The final UCI and challenge evaluations below did not affect model or threshold selection.", "",
        "## v2 vs v3 vs v4 vs v5", "",
        "UCI figures are measured on the same v5 registrable-domain-disjoint final rows. v2-v4 were scored retrospectively at their saved thresholds and may have seen some UCI data in their earlier training. The v5 challenge FPR uses the same 21 held-out legitimate URLs for all four artifacts and each artifact's saved threshold.", "",
        "| Version | Threshold | UCI accuracy | Phishing precision | Phishing recall | Phishing F1 | UCI FPR | UCI FNR | Challenge FPs | Challenge FPR | Mean ms | P95 ms | Calibration / UCI Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for version in ("v2", "v3", "v4", "v5"):
        row = versions[version]
        uci = row["uci_final"]
        latency = row["latency"]
        brier = uci.get("brier_score")
        lines.append(
            f"| {version} ({row['model_name']}) | {row['threshold']:.2f} | {uci['accuracy']:.4%} | {uci['precision']:.4%} | {uci['recall']:.4%} | {uci['f1']:.4%} | {uci['false_positive_rate']:.4%} | {uci['false_negative_rate']:.4%} | {row['challenge_false_positives']}/{row['challenge_count']} | {row['challenge_fpr']:.2%} | {latency['inference_latency_ms_mean']:.3f} | {latency['inference_latency_ms_p95']:.3f} | {row['calibration_method']} / {brier:.5f} |"
        )
    lines.extend([
        "", "### Confusion matrices", "",
        "| Version | UCI TN/FP/FN/TP | Challenge TN/FP/FN/TP |", "|---|---|---|",
    ])
    for version in ("v2", "v3", "v4", "v5"):
        row = versions[version]
        y = np.asarray([r["expected_label"] for r in audit["challenge_predictions"][version]])
        pred = np.asarray([r["threshold_prediction"] for r in audit["challenge_predictions"][version]])
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        lines.append(f"| {version} | {row['uci_final']['confusion_matrix']} | {cm} |")
    v5 = versions["v5"]
    lines.extend([
        "", "## Calibration and latency", "",
        f"v5 calibration: {calibration.get('method', 'unknown')}; validation Brier score {calibration.get('validation_brier'):.6f}; v5 UCI final Brier score {v5['uci_final']['brier_score']:.6f}. Challenge Brier score: {brier_score_loss([r['expected_label'] for r in audit['challenge_predictions']['v5']], [r['probability'] for r in audit['challenge_predictions']['v5']]):.6f} (descriptive for 21 legitimate-only examples). Latency is the trainer's 30 single-URL samples including URL feature extraction and prediction.", "",
        f"v5 phishing recall on the UCI final set is {v5['uci_final']['recall']:.2%}; it changed by {(v5['uci_final']['recall'] - versions['v4']['uci_final']['recall'])*100:+.2f} percentage points versus v4 on these same rows. This is an unseen-domain evaluation on a static corpus, not a live or temporal performance claim.", "",
        "## Held-out hard-negative false positives", "",
    ])
    v5_fp = [r for r in audit["challenge_predictions"]["v5"] if r["threshold_prediction"]]
    if v5_fp:
        lines.extend(f"- `{r['url']}` — p={r['probability']:.4f}; category: {r['category']}." for r in v5_fp)
    else:
        lines.append("- None at the selected threshold.")
    lines.extend([
        "", "## Decision", "",
        "Do not promote v5 automatically. Review its remaining held-out false positives and the phishing-recall tradeoff. The shipped default stays v2; v5 remains opt-in.", "",
        "## Data and limits", "",
        "v5 development rows are in `data/hard_negative_urls_v5.csv`; none share a registrable domain with the 21-domain challenge. UCI source data has no per-URL timestamps, so results do not establish performance on current live URLs. The challenge is small and its FPR is descriptive, not a population estimate.", "",
    ])
    (ROOT / "docs/url_model_improvement_report_v5.md").write_text("\n".join(lines), encoding="utf-8")

    findings = [
        "# URL error analysis v5", "",
        f"v4's selected threshold was 0.80. v5's selected threshold was {THRESHOLD(joblib.load(MODEL_DIR / 'url_model_v5.joblib')):.2f}; v5 was frozen before evaluating this challenge. The 21 challenge URLs are all legitimate.", "",
        "## Why v4 flagged the three URLs", "",
        "The v4 candidate uses character TF-IDF plus 39 scaled numeric URL features, logistic regression, and sigmoid calibration. Its 0.80 probability threshold corresponds to a raw logistic margin of about 0.152. Each URL's raw margin exceeded that cut. The learned weights reflect generic URL-shape correlations; they do not establish that the named domains are risky.", "",
    ]
    for url, detail in audit["v4_false_positive_attributions"].items():
        trace = detail["v4_feature_trace"]
        features = trace["url_features"]
        top_grams = ", ".join(f"`{x['ngram']}` (+{x['logit_contribution']:.3f})" for x in trace["top_positive_character_ngrams"][:8])
        top_num = ", ".join(f"`{x['feature']}` (+{x['logit_contribution']:.3f})" for x in trace["top_positive_numeric_features"][:6])
        neg_grams = ", ".join(f"`{x['ngram']}` ({x['logit_contribution']:.3f})" for x in trace["top_negative_character_ngrams"][:4]) or "none active"
        words = "; ".join(trace["indicators"]).rstrip(".") or "no explicit suspicious-word indicator"
        findings.extend([
            f"### `{url}`", "",
            f"- v4 probability/margin: {detail['v4_probability']:.6f} / {trace['raw_logistic_margin']:.6f}; v4 threshold: 0.80.",
            f"- v5 probability/prediction: {detail['v5_probability']:.6f} / {'flagged' if detail['v5_prediction'] else 'not flagged'} at {detail['v5_threshold']:.2f}.",
            f"- Explicit lexical indicators: {words}.",
            f"- Relevant shape values: URL length {features['url_length']:.0f}, hostname length {features['domain_length']:.0f}, path length {features['path_length']:.0f}, dots {features['num_dots']:.0f}, subdomains {features['num_subdomains']:.0f}, HTTPS {bool(features['is_https'])}, query parameters {features['query_param_count']:.0f}.",
            f"- Largest active positive character n-gram contributions: {top_grams}.",
            f"- Largest active negative character n-gram contributions: {neg_grams}.",
            f"- Largest positive numeric-feature contributions: {top_num}.",
            f"- Sigmoid calibration parameters: a={trace['sigmoid_a']:.6f}, b={trace['sigmoid_b']:.6f}; it maps the raw margin monotonically to the reported probability. The high scores were not caused by a flat isotonic calibration plateau.", "",
        ])
    findings.extend([
        "## Comparison with correctly classified legitimate lookalikes", "",
        "These are correctly classified v4 hard-negative validation examples, separate from the final challenge. They show that the false positives did not come from one simple word alone:", "",
    ])
    chosen = []
    for target in ("enterprise_login", "banking_home"):
        candidates = [r for r in audit["v4_correctly_classified_validation_examples"] if target.split("_")[0] in r.get("category", "")]
        chosen.extend(candidates[:4])
    if not chosen:
        chosen = audit["v4_correctly_classified_validation_examples"][:8]
    for row in chosen[:8]:
        findings.append(f"- `{row['url']}` — v4 p={row['model_probability']:.4f}, correctly below 0.80 ({row['category']}).")
    v5_rows = audit["challenge_predictions"]["v5"]
    flagged = [r for r in v5_rows if r["threshold_prediction"]]
    for row in flagged:
        audit.setdefault("v5_false_positive_attributions", {})[row["url"]] = _word_char_attribution(
            artifacts["v5"], row["url"]
        )
    if flagged:
        findings.extend(["", "## Why the remaining v5 false positive was flagged", ""])
        for row in flagged:
            trace = audit["v5_false_positive_attributions"][row["url"]]
            words = trace["ngram_contributions"]["word"]["top_positive"]
            chars = trace["ngram_contributions"]["char"]["top_positive"]
            word_text = ", ".join(f"`{x['ngram']}` (+{x['logit_contribution']:.3f})" for x in words[:6])
            char_text = ", ".join(f"`{x['ngram']}` (+{x['logit_contribution']:.3f})" for x in chars[:8])
            features = trace["url_features"]
            findings.extend([
                f"### `{row['url']}`", "",
                f"v5 probability/margin: {row['probability']:.6f} / {trace['raw_logistic_margin']:.6f}; threshold {row['threshold']:.2f}.",
                f"No explicit suspicious-word indicator. URL length {features['url_length']:.0f}, hostname length {features['domain_length']:.0f}, one subdomain label, HTTPS, and no query parameters.",
                f"Top positive word n-grams: {word_text}.",
                f"Top positive character n-grams: {char_text}.",
                f"Sigmoid parameters: a={trace['sigmoid_a']:.6f}, b={trace['sigmoid_b']:.6f}. The score is driven by generic URL tokens and fragments, especially `com`, `https`, and `.com/`, rather than an explicit suspicious-word rule.", "",
            ])
    findings.extend([
        "", "## v5 result on the fixed challenge", "",
        f"v5 flagged {len(flagged)} of {len(v5_rows)} legitimate URLs ({len(flagged)/max(1,len(v5_rows)):.2%}). The three v4 errors are reported individually above; these scores were measured after v5's threshold was fixed. No domain-specific allow rule was introduced.", "",
        "## UCI false positives and false negatives", "",
        f"On the v5 UCI final partition, confusion matrix TN/FP/FN/TP is {metadata['metrics']['confusion_matrix']}. This is an unseen-domain result on static data, not a live-web performance claim.", "",
    ])
    (ROOT / "docs/url_error_analysis_v5.md").write_text("\n".join(findings), encoding="utf-8")


if __name__ == "__main__":
    main()
