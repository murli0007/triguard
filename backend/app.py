from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from domain_utils import detect_brand_impersonation, is_verified_trusted_domain
from email_features import clean_email_text, email_indicators
from malware_features import FEATURE_NAMES as MALWARE_FEATURES, MAX_FILE_BYTES, SUPPORTED_EXTENSIONS, extract_malware_features
from risk_engine import SAFE_MAX, SUSPICIOUS_MAX, classify_result, combine_scores, label_url_probability, url_risk_decision, url_thresholds
from url_features import (FEATURE_NAMES as URL_FEATURES,
                          canonicalize_root_path_for_model,
                          extract_url_features, normalize_url, url_indicators,
                          url_signal_count)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
PORT = 5050
MODE = os.getenv("APP_MODE", "real" if (MODEL_DIR / "url_model.joblib").exists() else "demo").lower()
if MODE not in {"demo", "real"}:
    raise RuntimeError("APP_MODE must be 'demo' or 'real'")
MAX_REQUEST_BYTES = MAX_FILE_BYTES + 64 * 1024

app = Flask(__name__, static_folder=str(PROJECT_DIR / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
if os.getenv("ENABLE_EXTENSION_CORS", "1") == "1":
    CORS(app, resources={r"/api/*": {"origins": [r"chrome-extension://.*", "http://127.0.0.1:5050", "http://localhost:5050"]}})
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("triguard")
_artifacts: dict[str, dict] = {}


def model_path(name: str) -> Path:
    if MODE == "real":
        ver = os.getenv("URL_MODEL_VERSION", "v6").lower()
        if name == "url" and ver in {"v3", "v4", "v5", "v6"}:
            return MODEL_DIR / f"url_model_{ver}.joblib"
        return MODEL_DIR / f"{name}_model.joblib"
    return MODEL_DIR / f"{name}_model_demo.joblib"


def get_artifact(name: str) -> dict:
    if name not in _artifacts:
        path = model_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"{name} model is unavailable in {MODE.upper()} mode; train it first")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise RuntimeError(f"Invalid {name} model artifact")
        artifact.setdefault("training_source", "unknown")
        artifact["artifact_path"] = str(path)
        _artifacts[name] = artifact
    return _artifacts[name]


def safe_predict(name: str, vector, features: dict, expected_names: list[str]):
    artifact = get_artifact(name)
    names = artifact.get("feature_names", expected_names)
    if not isinstance(names, list) or any(feature not in expected_names for feature in names):
        raise RuntimeError(f"{name} model feature schema does not match current extractor; retrain this model")
    # Pandas columns are identical during fit/inference and eliminate sklearn feature-name warnings.
    import pandas as pd
    row = pd.DataFrame([[features[key] for key in names]], columns=names)
    if name == "url" and artifact.get("uses_raw_url_text"):
        row.insert(0, "url", str(vector))
    model = artifact["model"]
    probabilities = model.predict_proba(row)[0]
    class_risks = artifact.get("risk_classes", {"malicious": 1})
    risk_class = int(class_risks.get("malicious", 1))
    by_class = {int(label): float(prob) for label, prob in zip(model.classes_, probabilities)}
    risk = round(100 * by_class.get(risk_class, 0.0), 2)
    malicious_probability = by_class.get(risk_class, 0.0)
    return artifact, risk, malicious_probability


def log_inference(module: str, started: float, prediction: str, score: float) -> None:
    logger.info("inference module=%s duration_ms=%.2f prediction=%s score=%.2f", module, (time.perf_counter() - started) * 1000, prediction, score)


def record_reasons(module: str, features: dict, prediction: str, model_name: str) -> list[str]:
    reasons = [f"{model_name} model classified this input as {prediction}."]
    if module == "url":
        if features["is_ip_host"]:
            reasons.append("The hostname is an IP address.")
        if features["has_punycode"]:
            reasons.append("The hostname uses punycode encoding.")
        if features["suspicious_word_count"]:
            reasons.append("The URL contains security or account related terms.")
        if features["num_at"]:
            reasons.append("The URL contains an @ character that can obscure the destination.")
        if features["is_https"] == 0:
            reasons.append("The URL does not use HTTPS.")
        if len(reasons) == 1:
            reasons.append("No high-signal lexical URL indicators were found.")
    if module == "file":
        if features["pe_magic"]:
            reasons.append("The file begins with a Windows PE executable marker.")
        if features["entropy"] > 7.2:
            reasons.append("The sampled bytes have high Shannon entropy.")
        if features["suspicious_token_count"]:
            reasons.append("Static strings include potentially suspicious command or API terms.")
        if len(reasons) == 1:
            reasons.append("Static file metadata and byte features informed the result.")
    return reasons


@app.get("/")
def index():
    return send_from_directory(PROJECT_DIR / "frontend", "index.html")


@app.get("/api/health")
def health():
    models = {}
    metadata_by_name = {}
    url_ver = os.getenv("URL_MODEL_VERSION", "v6").lower()
    metadata_names = {
        "url": (
            "url_demo_training_metrics.json"
            if MODE == "demo"
            else (
                f"url_model_{url_ver}_metadata.json"
                if url_ver in {"v3", "v4", "v5", "v6"}
                else "url_model_metadata.json"
            )
        ),
        "email": "email_demo_training_metrics.json" if MODE == "demo" else "email_metadata.json",
        "malware": "malware_demo_training_metrics.json" if MODE == "demo" else "malware_metadata.json",
    }
    for name in ("url", "email", "malware"):
        path = model_path(name)
        metadata_path = MODEL_DIR / metadata_names[name]
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        metadata_by_name[name] = metadata
        models[name] = {
            "available": path.is_file(),
            "version": metadata.get("version", "unknown"),
            "dataset": metadata.get("dataset", "demo" if MODE == "demo" else "not trained"),
            "training_source": metadata.get("training_source", "demo" if MODE == "demo" else "unknown"),
        }
    try:
        selected_url_thresholds = url_thresholds(metadata_by_name["url"])
    except ValueError:
        selected_url_thresholds = {"suspicious_probability": 0.30, "malicious_probability": 0.70}
    return jsonify({"status": "ok", "port": PORT, "mode": MODE, "models": models,
                    "risk_thresholds": {"safe_max": SAFE_MAX, "suspicious_max": SUSPICIOUS_MAX},
                    "url_thresholds": selected_url_thresholds})


@app.post("/api/check-url")
def check_url():
    started = time.perf_counter()
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("url"), str):
        return jsonify({"error": "JSON body with string field 'url' is required"}), 400
    try:
        url = normalize_url(data["url"])
        selected_artifact = get_artifact("url")
        model_url = canonicalize_root_path_for_model(url) if selected_artifact.get("canonicalize_root_slash") else url
        features = extract_url_features(model_url)
        artifact, risk, probability = safe_predict("url", model_url, features, URL_FEATURES)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("URL inference failed")
        return jsonify({"error": "URL analysis is temporarily unavailable"}), 500
    thresholds = url_thresholds(artifact)
    model_risk = risk

    trusted, trusted_reason = is_verified_trusted_domain(url, features)
    impersonation, imp_reason = detect_brand_impersonation(url, features)

    structural_signals = url_signal_count(features)
    indicators = url_indicators(features, url)

    if trusted:
        probability = min(probability, 0.05)
        model_risk = min(model_risk, 5.0)
        prediction, risk = "safe", min(risk, 15.0)
        model_risk = max(model_risk, risk)
        reasons = [trusted_reason]
        if indicators:
            reasons.extend(indicators)
    elif impersonation:
        probability = max(probability, 0.98)
        model_risk = 100.0
        structural_signals += 2
        indicators.append(imp_reason)
        prediction, risk = url_risk_decision(probability, thresholds, structural_signals)
        reasons = [imp_reason, f"URL model classified this URL as {label_url_probability(probability, thresholds)}.", *indicators]
    else:
        prediction, risk = url_risk_decision(probability, thresholds, structural_signals)
        reasons = [f"URL model classified this URL as {label_url_probability(probability, thresholds)}.", *indicators]
        if prediction == "suspicious" and model_risk >= thresholds["malicious_probability"] * 100:
            reasons.append("Severity capped at SUSPICIOUS because fewer than two independent high-risk URL structure signals support the model score.")
        if not indicators:
            reasons.append("No listed lexical indicator was present; this is a model-only classification from the combined URL feature vector.")
    result = classify_result(risk, {"url": model_risk}, probability, artifact.get("probability_calibrated", False), reasons, indicators,
        "Sigmoid-calibrated estimate on a domain-disjoint validation set; not a guaranteed probability of attack." if artifact.get("probability_calibrated") else "Uncalibrated DEMO model output; not a guaranteed probability of attack.")
    result.update({"type": "url", "url": url, "prediction": prediction, "model": artifact.get("model_name", "URL lexical classifier"), "model_version": artifact.get("model_version", "unknown"), "training_source": artifact["training_source"], "model_source": artifact["training_source"], "decision_thresholds": thresholds})
    log_inference("url", started, prediction, risk)
    return jsonify(result)


@app.post("/api/check-email")
def check_email():
    started = time.perf_counter()
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON object with subject and body fields is required"}), 400
    subject, body = data.get("subject", ""), data.get("body", "")
    if not isinstance(subject, str) or not isinstance(body, str):
        return jsonify({"error": "subject and body must be strings"}), 400
    if len(subject) > 500 or len(body) > 200_000:
        return jsonify({"error": "email exceeds input size limits"}), 413
    text = clean_email_text(subject, body)
    if not text:
        return jsonify({"error": "subject or body is required"}), 400
    try:
        artifact = get_artifact("email")
        model = artifact["model"]
        probs = model.predict_proba([text])[0]
        risk_class = int(artifact.get("risk_classes", {}).get("malicious", 1))
        by_class = {int(label): float(prob) for label, prob in zip(model.classes_, probs)}
        risk = round(100 * by_class.get(risk_class, 0.0), 2)
        email_probability = float(by_class.get(risk_class, 0.0))
        email_calibrated = bool(artifact.get("probability_calibrated", False))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("Email inference failed")
        return jsonify({"error": "Email analysis is temporarily unavailable"}), 500
    prediction = __import__("risk_engine").label_from_score(risk)
    indicators = email_indicators(subject, body)
    model_scores = {"email": risk}
    # Very short, indicator-free text carries little evidence for a language
    # classifier. Avoid turning a generic two-word note into a phishing alert
    # solely because those words correlate with a training collection.
    short_low_evidence = len(set(text.split())) <= 3 and not indicators
    if short_low_evidence:
        risk = min(risk, float(SAFE_MAX))
        model_scores["email"] = risk
        reasons = ["Message is too short to support a reliable phishing classification; no configured threat indicators were found."]
    else:
        reasons = []
    link_scores = []
    for link in dict.fromkeys(re.findall(r"(?:https?://|www\.)[^\s<>\"']+", f"{subject} {body}", re.I)):
        try:
            normalized_link = normalize_url(link)
            selected_url_artifact = get_artifact("url")
            model_link = canonicalize_root_path_for_model(normalized_link) if selected_url_artifact.get("canonicalize_root_slash") else normalized_link
            link_features = extract_url_features(model_link)
            link_artifact, link_model_risk, link_probability = safe_predict("url", model_link, link_features, URL_FEATURES)
            link_thresholds = url_thresholds(link_artifact)
            link_trusted, _ = is_verified_trusted_domain(normalized_link, link_features)
            link_imp, link_imp_reason = detect_brand_impersonation(normalized_link, link_features)
            if link_trusted:
                link_risk = min(link_model_risk, 10.0)
            elif link_imp:
                link_risk = 95.0
                indicators.append(link_imp_reason)
            else:
                signals = url_signal_count(link_features)
                _, link_model_risk = url_risk_decision(link_probability, link_thresholds, signals)
                link_risk = link_model_risk
            link_scores.append(link_risk)
        except (ValueError, FileNotFoundError, RuntimeError):
            continue
    if link_scores:
        model_scores["embedded_urls"] = max(link_scores)
        risk = combine_scores(list(model_scores.values()))
        indicators.append(f"Embedded URL model found a maximum risk score of {max(link_scores):.0f}/100.")
    prediction = __import__("risk_engine").label_from_score(risk)
    if not short_low_evidence:
        reasons = [f"Email language model classified this message as {__import__('risk_engine').label_from_score(model_scores['email'])}.", *indicators]
    if not indicators:
        reasons.append("No configured supporting email indicators were found.")
    # Email body stays in memory for this request and is never included in logs/results.
    reported_probability = None if short_low_evidence else email_probability
    probability_note = ("Insufficient text for a reliable email-model probability." if short_low_evidence else
        "Calibrated email-model estimate; not a guaranteed probability of attack." if email_calibrated else
        "Uncalibrated DEMO model output; not a guaranteed probability of attack.")
    result = classify_result(risk, model_scores, reported_probability, email_calibrated, reasons, indicators, probability_note)
    result["prediction"] = prediction
    result.update({"type": "email", "model": artifact.get("model_name", "Email classifier") + (" + URL model" if link_scores else ""), "model_version": artifact.get("model_version", "unknown"), "training_source": artifact.get("training_source", "unknown"), "model_source": artifact.get("training_source", "unknown")})
    log_inference("email", started, prediction, risk)
    return jsonify(result)


@app.post("/api/check-file")
def check_file():
    started = time.perf_counter()
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "file is required"}), 400
    filename = secure_filename(uploaded.filename)
    extension = Path(filename).suffix.lower()
    if not filename or extension not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"unsupported file type; supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"}), 415
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="static-scan-", suffix=extension, delete=False) as temporary:
            tmp_path = Path(temporary.name)
            total = 0
            while chunk := uploaded.stream.read(64 * 1024):
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise ValueError("file exceeds the 10 MiB limit")
                temporary.write(chunk)
        features = extract_malware_features(tmp_path)
        artifact, risk, confidence = safe_predict("malware", None, features, MALWARE_FEATURES)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 413 if "limit" in str(exc) else 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        logger.exception("File inference failed")
        return jsonify({"error": "File analysis is temporarily unavailable"}), 500
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
    prediction = __import__("risk_engine").label_from_score(risk)
    reasons = record_reasons("file", features, prediction, "Static malware")
    probability = confidence
    result = classify_result(risk, {"malware": risk}, probability, bool(artifact.get("probability_calibrated", False)), reasons, reasons[1:],
        "Calibrated estimate." if artifact.get("probability_calibrated") else "Uncalibrated static-model output; not a guaranteed probability of attack.")
    result.update({"type": "file", "filename": filename, "model": artifact.get("model_name", "Static feature Random Forest"), "model_version": artifact.get("model_version", "unknown"), "training_source": artifact.get("training_source", "unknown"), "model_source": artifact.get("training_source", "unknown")})
    log_inference("file", started, prediction, risk)
    return jsonify(result)


@app.post("/api/check-multi")
def check_multi():
    """Combine already computed, explicitly relevant detector scores."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("model_scores"), dict) or not data["model_scores"]:
        return jsonify({"error": "model_scores must be a non-empty object of detector scores"}), 400
    try:
        scores = {str(name): max(0, min(100, float(score))) for name, score in data["model_scores"].items()}
    except (TypeError, ValueError):
        return jsonify({"error": "model scores must be numeric values from 0 to 100"}), 400
    score = combine_scores(list(scores.values()))
    result = classify_result(score, scores, None, False, ["Combined from the supplied relevant detector results."], probability_note="Combined application score; no calibrated combined probability is available.")
    return jsonify(result)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "request exceeds the configured size limit"}), 413


if __name__ == "__main__":
    logger.info("Starting %s mode; models in %s", MODE.upper(), MODEL_DIR)
    app.run(host="127.0.0.1", port=PORT, debug=os.getenv("FLASK_DEBUG", "0") == "1")
