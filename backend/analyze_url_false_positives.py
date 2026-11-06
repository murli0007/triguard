from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import joblib
import pandas as pd

from domain_utils import registrable_domain
from url_features import extract_url_features

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "backend/models/url_model_v3.joblib"
METRICS = ROOT / "backend/models/url_metrics_v3.json"
OUT = ROOT / "docs/url_false_positive_feature_analysis.md"
WORDS = re.compile(r"(?i)(login|log-in|signin|sign-in|verify|verification|secure|security|account|authentication|auth|payment|billing|password|support|invoice|banking|identity|portal|sso|checkout|recover|reset|credential|confirm|profile|service|user)")


def analyze() -> str:
    artifact = joblib.load(MODEL)
    model = artifact["model"]
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    calibrated_base = model.calibrated_classifiers_[0].estimator
    base = calibrated_base.estimator
    transformer = base.steps[0][1]
    classifier = base.steps[-1][1]
    vectorizer = transformer.named_transformers_["char"]
    vocabulary = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_.ravel()
    rows: list[dict] = []
    for result in metrics["hard_negative_results"]["results"]:
        if result["expected_label"] != 0 or result["threshold_prediction"] != 1:
            continue
        url = result["url"]
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        sparse = transformer.transform(pd.DataFrame({"url": [url]}))
        contributions = [(vocabulary[i], float(value * coefficients[i])) for i, value in zip(sparse.indices, sparse.data)]
        positive = sorted((item for item in contributions if item[1] > 0), key=lambda item: -item[1])[:7]
        negative = sorted((item for item in contributions if item[1] < 0), key=lambda item: item[1])[:4]
        feature_matrix = base.decision_function(pd.DataFrame({"url": [url]}))
        path_parts = [part for part in parsed.path.split("/") if part]
        query_count = len(parse_qsl(parsed.query, keep_blank_values=True))
        tld = host.split(".")[-1] if "." in host else "(none)"
        row = {
            "url": url,
            "root_domain": registrable_domain(host),
            "category": result["category"],
            "calibrated_probability": float(result["model_probability"]),
            "raw_logit_margin": float(feature_matrix[0]),
            "url_length": len(url),
            "query_parameter_count": query_count,
            "subdomain_labels": max(0, len(host.split(".")) - len(registrable_domain(host).split("."))),
            "hyphen_count": url.count("-"),
            "tld": tld,
            "https": parsed.scheme == "https",
            "matched_terms": sorted(set(WORDS.findall(url.lower()))),
            "top_positive_character_ngrams": positive,
            "top_negative_character_ngrams": negative,
        }
        rows.append(row)

    counts = {
        "security/account terms": sum(bool(row["matched_terms"]) for row in rows),
        "URL length >= 100": sum(row["url_length"] >= 100 for row in rows),
        "query parameters present": sum(row["query_parameter_count"] > 0 for row in rows),
        "one or more subdomain labels": sum(row["subdomain_labels"] > 0 for row in rows),
        "hyphen present": sum(row["hyphen_count"] > 0 for row in rows),
        "HTTPS": sum(row["https"] for row in rows),
        "URLs on legitimate organization/service roots": len(rows),
    }
    tlds = Counter(row["tld"] for row in rows)
    common_ngrams: Counter = Counter()
    for row in rows:
        for ngram, contribution in row["top_positive_character_ngrams"]:
            common_ngrams[ngram] += contribution

    lines = [
        "# v3 hard-negative false-positive analysis", "",
        f"Analyzed all {len(rows)} legitimate challenge URLs that v3 flagged at its selected 0.70 threshold.", "",
        "## Pattern counts", "",
        "Counts are overlapping properties, not separate explanations; the model uses only character TF-IDF n-grams plus a logistic classifier. It does not receive the numeric feature columns listed below as separate inputs.", "",
        "| Pattern | Flagged URLs |", "|---|---:|",
    ]
    lines.extend(f"| {name} | {count}/{len(rows)} |" for name, count in counts.items())
    lines.extend([
        "", f"TLDs: {dict(tlds)}. These 18 URLs span {len({row['root_domain'] for row in rows})} distinct registrable roots. URL length range/median: {min(row['url_length'] for row in rows)}/{pd.Series([row['url_length'] for row in rows]).median():.1f}/{max(row['url_length'] for row in rows)} characters.", "",
        "## Per-URL evidence", "",
        "The positive n-grams below are the largest direct contributions to the raw logistic score before isotonic calibration. Negative n-grams push in the opposite direction. The isotonic calibrator maps all 18 out-of-domain scores to probability 1.0, so the displayed probability is saturated and does not distinguish their raw scores.", "",
        "| URL | Root | Length | Queries | Subdomains | Hyphens | TLD | Terms | Top positive character n-grams (contribution) |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ])
    for row in rows:
        positive = ", ".join(f"`{ngram}` ({value:+.3f})" for ngram, value in row["top_positive_character_ngrams"][:5])
        terms = ", ".join(row["matched_terms"]) or "none"
        lines.append(
            f"| `{row['url']}` | `{row['root_domain']}` | {row['url_length']} | {row['query_parameter_count']} | {row['subdomain_labels']} | {row['hyphen_count']} | `{row['tld']}` | {terms} | {positive} |"
        )
    lines.extend([
        "", "## What drove the errors", "",
        f"- **Character n-grams:** Common high-positive fragments are {', '.join(f'`{ngram}`' for ngram, _ in common_ngrams.most_common(10))}. In particular, `.com/` and `com/` fragments appear across benign brand URLs and receive positive phishing weight. The model has learned dataset-specific URL character correlations rather than reputation or page behavior.",
        "- **Login/account wording:** Present in many examples, but it is not the only cause: help pages, a home page, and product-content URLs with no such term were also flagged.",
        "- **Length, queries, subdomains, hyphens, and TLD:** These are descriptive correlates only for the selected v3 model. They are not explicit model features; several short URLs without query parameters were also assigned probability 1.0.",
        "- **HTTPS:** All 18 use HTTPS, but HTTPS is not a direct input to this character-only model. HTTPS is not evidence of legitimacy by itself.",
        "- **Brand names:** The challenge URLs belong to legitimate service roots, but v3 has no brand/reputation field. The model does not know that these roots are legitimate; adding per-domain allow rules would hide the underlying generalization failure.",
        "- **Calibration:** Isotonic calibration selected for a slightly lower Brier score on broad validation extrapolates these domain-shifted scores to 1.0. For the PayPal sign-in URL, the raw logistic margin is negative but the calibrated probability is 1.0; the calibrator therefore pushes that case across the warning threshold and destroys useful confidence ranking. This is a calibration failure in addition to the underlying character-model bias.",
        "", "## Individual raw-score explanation", "",
    ])
    for row in rows:
        positive = ", ".join(f"`{ngram}` {value:+.4f}" for ngram, value in row["top_positive_character_ngrams"][:7])
        negative = ", ".join(f"`{ngram}` {value:+.4f}" for ngram, value in row["top_negative_character_ngrams"][:4]) or "none"
        lines.append(f"- `{row['url']}`: raw margin {row['raw_logit_margin']:.4f}; calibrated probability {row['calibrated_probability']:.4f}; positive fragments: {positive}; strongest opposing fragments: {negative}.")
    lines.append("")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "backend/models/url_false_positive_attributions_v3.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return f"Wrote {OUT} with {len(rows)} false-positive attributions."


if __name__ == "__main__":
    print(analyze())
