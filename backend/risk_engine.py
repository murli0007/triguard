from __future__ import annotations

import os

SAFE_MAX = int(os.getenv("RISK_SAFE_MAX", "29"))
SUSPICIOUS_MAX = int(os.getenv("RISK_SUSPICIOUS_MAX", "69"))
if not 0 <= SAFE_MAX < SUSPICIOUS_MAX < 100:
    raise ValueError("Risk thresholds must satisfy 0 <= RISK_SAFE_MAX < RISK_SUSPICIOUS_MAX < 100")


def label_from_score(score: float) -> str:
    """Generic score bands for modules without validation-tuned thresholds."""
    if score >= SUSPICIOUS_MAX + 1:
        return "malicious"
    if score > SAFE_MAX:
        return "suspicious"
    return "safe"


def url_thresholds(artifact: dict | None = None) -> dict[str, float]:
    """Use thresholds selected on URL validation data, with explicit env overrides."""
    saved = (artifact or {}).get("decision_thresholds", {})
    # Candidate artifacts carry thresholds selected on leakage-safe validation;
    # v2's saved default is 0.70 to reduce warnings from moderate lexical scores.
    suspicious = float(os.getenv("URL_SUSPICIOUS_THRESHOLD", saved.get("suspicious_probability", 0.70)))
    malicious = float(os.getenv("URL_MALICIOUS_THRESHOLD", saved.get("malicious_probability", 0.70)))
    if not 0 <= suspicious <= malicious <= 1:
        raise ValueError("URL probability thresholds must satisfy 0 <= suspicious <= malicious <= 1")
    return {"suspicious_probability": suspicious, "malicious_probability": malicious}


def label_url_probability(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds["malicious_probability"]:
        return "malicious"
    if probability >= thresholds["suspicious_probability"]:
        return "suspicious"
    return "safe"


def url_risk_decision(probability: float, thresholds: dict[str, float], structural_signals: int) -> tuple[str, float]:
    """Convert URL probability to triage risk; weakly supported extreme scores are downgraded."""
    prediction = label_url_probability(probability, thresholds)
    score = round(max(0.0, min(1.0, float(probability))) * 100, 2)
    if prediction == "safe":
        return "safe", min(score, float(SAFE_MAX))
    if prediction == "suspicious":
        return "suspicious", min(score, float(SUSPICIOUS_MAX))
    if structural_signals < 2:
        return "suspicious", min(score, float(SUSPICIOUS_MAX))
    return prediction, score


def combine_scores(scores: list[float]) -> float:
    """Combine only supplied, relevant 0-100 application scores."""
    valid = [max(0.0, min(100.0, float(score))) for score in scores]
    if not valid:
        return 0.0
    if len(valid) == 1:
        return round(valid[0], 2)
    strongest_index = max(range(len(valid)), key=valid.__getitem__)
    strongest = valid[strongest_index]
    others = valid[:strongest_index] + valid[strongest_index + 1:]
    return round(min(100.0, 0.8 * strongest + 0.2 * (sum(others) / len(others))), 2)


def classify_result(score: float, model_scores: dict[str, float], model_probability: float | None,
                    probability_calibrated: bool, reasons: list[str],
                    indicators: list[str] | None = None, probability_note: str | None = None) -> dict:
    return {
        "risk_score": round(float(score), 2),
        "prediction": label_from_score(score),
        "model_probability": round(float(model_probability), 6) if model_probability is not None else None,
        "probability_calibrated": bool(probability_calibrated),
        "probability_note": probability_note or "A model estimate, not a guaranteed probability of a real-world attack.",
        "model_scores": model_scores,
        "reasons": list(dict.fromkeys(reasons)),
        "indicators": indicators or [],
    }
